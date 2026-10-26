"""TAPNext++ runtime: the same Backend interface as trackon_runtime (step / remove /
n / name / gpu_ms), so trackon_runtime.PointTracker and run_video.py work with
either model.

State (see tapnextpp_export/step.py):
  video_state [12,4,4096,768]  kept on the device, ping-pong between frames
  rows        [1+N,12,4,768]   row 0 = prototype (`unknown` token every frame),
                               rows 1..N = the points, in the order they were added
Adding k points appends k rows that start from a copy of the prototype's state
with the query token (= the official model with a late query); removing points
deletes their rows. Nothing is re-initialised.

When there are no points the model is not run at all (zero cost); the next
frame that has points starts a new sequence (video_reset = 1), exactly like the
official tracker's first frame.

Backends:
  torch     TapNextPPStep (the verified one-frame wrapper of the official model)
  ort-cpu   onnxruntime CPU
  ort-cuda  onnxruntime-gpu CUDA EP
  trt       TensorRT engine from tools/build_tapnextpp_engine.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from trackon_runtime import import_tensorrt, load_cuda_dlls_for_ort  # pylint: disable=g-import-not-at-top

L, P, C = 12, 4, 768
TOK_MASK, TOK_QUERY, TOK_UNKNOWN = 0.0, 1.0, 2.0
IN_NAMES = ['frame', 'video_state', 'video_reset', 'point_state', 'point_ctl']
OUT_NAMES = ['point_out', 'video_state_out', 'point_state_out']
DEFAULT_RES = 512


class TapNextBackend:
  """Row bookkeeping; subclasses implement _append(k, copy_prototype), _keep(idx), _run(...)."""

  name = 'tapnext++'
  delta_v = 0.5   # visible = sigmoid(logit) > 0.5, as the official tracker
  gpu_ms = None

  def __init__(self, resolution=DEFAULT_RES, new_points='prototype'):
    if new_points not in ('prototype', 'reset'):
      raise ValueError(new_points)
    self.res = resolution
    self.input_size = (resolution, resolution)
    self.num_video = (resolution // 8) ** 2
    self.new_points = new_points
    self._n = 0          # points (rows - 1)
    self._fresh = True   # next model run starts a new sequence

  @property
  def n(self):
    return self._n

  def reset(self):
    """Start a new sequence on the next step (all point states restart too)."""
    self._fresh = True

  def step(self, frame_rgb, new_xy):
    """frame_rgb [R,R,3] RGB (model input size); new_xy [k,2] model-input pixels.
    Returns points [N,2] (model-input px), vis_prob [N], certainty [N]; new points last."""
    new_xy = np.asarray(new_xy, np.float32).reshape(-1, 2)
    k = len(new_xy)
    if self._n == 0 and k == 0:  # nothing to track: skip the model
      self._fresh = True
      z = np.zeros(0, np.float32)
      return np.zeros((0, 2), np.float32), z, z
    first = self._fresh
    if k:
      self._append(k, copy_prototype=not first and self.new_points == 'prototype')
      self._n += k
    rows = self._n + 1
    ctl = np.zeros((rows, 4), np.float32)
    ctl[0, 2] = TOK_UNKNOWN
    if k:
      ctl[rows - k:, 0:2] = new_xy
      ctl[rows - k:, 2] = TOK_QUERY
      if self.new_points == 'reset':
        ctl[rows - k:, 3] = 1.0
    frame = np.ascontiguousarray(np.asarray(frame_rgb, np.float32).transpose(2, 0, 1)[None])
    out = self._run(frame, np.array([1.0 if first else 0.0], np.float32), ctl)
    self._fresh = False
    out = out[1:]
    return (np.ascontiguousarray(out[:, :2]), 1.0 / (1.0 + np.exp(-out[:, 2])),
            np.ascontiguousarray(out[:, 3]))

  def remove(self, rows):
    """Remove points by row index (rows of the last step's outputs)."""
    rows = set(int(r) for r in rows)
    if rows:
      keep = [0] + [i + 1 for i in range(self._n) if i not in rows]
      self._keep(np.array(keep, np.int64))
      self._n = len(keep) - 1


class TorchBackend(TapNextBackend):
  """TapNextPPStep in PyTorch (CPU or CUDA), state as device tensors."""

  def __init__(self, tapnet_root, weights, resolution=DEFAULT_RES, device=None, **kw):
    super().__init__(resolution, **kw)
    import torch  # pylint: disable=g-import-not-at-top
    from tapnextpp_export.step import TapNextPPStep, load_tapnextpp  # pylint: disable=g-import-not-at-top
    self.torch = torch
    self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    self.step_mod = TapNextPPStep(load_tapnextpp(tapnet_root, weights), resolution).eval().to(self.dev)
    self.name = f'tapnext++ torch {torch.__version__} {self.dev.type}'
    self.vs = torch.zeros(L, P, self.num_video, C, device=self.dev)
    self.ps = torch.zeros(1, L, P, C, device=self.dev)

  def _append(self, k, copy_prototype):
    t = self.torch
    src = self.ps[:1].expand(k, -1, -1, -1) if copy_prototype else t.zeros(k, L, P, C, device=self.dev)
    self.ps = t.cat([self.ps, src])

  def _keep(self, idx):
    self.ps = self.ps[self.torch.as_tensor(idx, device=self.dev)]

  def _run(self, frame, reset, ctl):
    t = self.torch
    with t.no_grad():
      out, self.vs, self.ps = self.step_mod(t.from_numpy(frame).to(self.dev), self.vs,
                                            t.from_numpy(reset).to(self.dev), self.ps,
                                            t.from_numpy(ctl).to(self.dev))
    return out.float().cpu().numpy()


class OrtBackend(TapNextBackend):
  """onnxruntime; the video state stays on the device, the point rows go to the
  host only when points are added or removed."""

  def __init__(self, onnx_path, device='cpu', **kw):
    import onnxruntime as ort  # pylint: disable=g-import-not-at-top
    if device == 'cuda':
      load_cuda_dlls_for_ort(ort)
      providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
      providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.sess = ort.InferenceSession(onnx_path, so, providers=providers)
    if device == 'cuda' and self.sess.get_providers()[0] != 'CUDAExecutionProvider':
      raise RuntimeError('CUDA EP not usable (run tools/check_env.py)')
    res = self.sess.get_inputs()[0].shape[-1]
    super().__init__(res, **kw)
    self.ort, self.device = ort, device
    self.name = f'tapnext++ ort-{device} ({self.sess.get_providers()[0]})'
    self.vs = ort.OrtValue.ortvalue_from_numpy(np.zeros((L, P, self.num_video, C), np.float32), device, 0)
    self.ps = self._up(np.zeros((1, L, P, C), np.float32))

  def _up(self, a):
    return self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a, np.float32), self.device, 0)

  def _append(self, k, copy_prototype):
    ps = self.ps.numpy()
    src = np.repeat(ps[:1], k, 0) if copy_prototype else np.zeros((k, L, P, C), np.float32)
    self.ps = self._up(np.concatenate([ps, src]))

  def _keep(self, idx):
    self.ps = self._up(self.ps.numpy()[idx])

  def _run(self, frame, reset, ctl):
    io = self.sess.io_binding()
    io.bind_cpu_input('frame', frame)
    io.bind_ortvalue_input('video_state', self.vs)
    io.bind_cpu_input('video_reset', reset)
    io.bind_ortvalue_input('point_state', self.ps)
    io.bind_cpu_input('point_ctl', ctl)
    io.bind_output('point_out', 'cpu')
    io.bind_output('video_state_out', self.device)
    io.bind_output('point_state_out', self.device)
    self.sess.run_with_iobinding(io)
    out, self.vs, self.ps = io.get_outputs()
    return out.numpy()


class TrtBackend(TapNextBackend):
  """TensorRT engine; all state on the GPU (torch tensors, ping-pong), row capacity grows x2."""

  def __init__(self, engine_path, **kw):
    trt = import_tensorrt()
    import torch  # pylint: disable=g-import-not-at-top
    self.trt, self.torch = trt, torch
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    self.engine = rt.deserialize_cuda_engine(open(engine_path, 'rb').read())
    if self.engine is None:
      raise RuntimeError('could not load the engine (built with another TensorRT version / GPU?)')
    self.ctx = self.engine.create_execution_context()
    res = tuple(self.engine.get_tensor_shape('frame'))[-1]
    super().__init__(res, **kw)
    prof = self.engine.get_tensor_profile_shape('point_state', 0)
    self.min_rows, self.max_rows = prof[0][0], prof[2][0]
    self.name = f'tapnext++ trt {trt.__version__} (points up to {self.max_rows - 1})'
    self.stream = torch.cuda.Stream()
    self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    cuda = lambda *s: torch.zeros(*s, dtype=torch.float32, device='cuda')
    self.frame = cuda(1, 3, res, res)
    self.reset_buf = cuda(1)
    self.vs = [cuda(L, P, self.num_video, C), cuda(L, P, self.num_video, C)]
    self.vping = 0
    self.cap = 0
    self._alloc(64)

  def _alloc(self, cap):
    t, rows = self.torch, self._n + 1
    cuda = lambda *s: t.zeros(*s, dtype=t.float32, device='cuda')
    new = dict(ps=[cuda(cap, L, P, C), cuda(cap, L, P, C)], ctl=cuda(cap, 4), out=cuda(cap, 4))
    if self.cap:
      new['ps'][0][:rows] = self.buf['ps'][self.ping][:rows]
    self.buf, self.cap, self.ping = new, cap, 0

  def _append(self, k, copy_prototype):
    rows = self._n + 1
    if rows + k > self.max_rows:
      raise RuntimeError(f'{rows + k - 1} points > engine max {self.max_rows - 1}: rebuild with a larger --max-points')
    if rows + k > self.cap:
      self._alloc(max(2 * self.cap, rows + k))
    ps = self.buf['ps'][self.ping]
    ps[rows:rows + k] = ps[0:1] if copy_prototype else 0

  def _keep(self, idx):
    ps = self.buf['ps'][self.ping]
    ps[:len(idx)] = ps[self.torch.as_tensor(idx, device='cuda')]

  def _run(self, frame, reset, ctl):
    t, b, p, rows = self.torch, self.buf, self.ping, len(ctl)
    ctx = self.ctx
    with t.cuda.stream(self.stream):
      self.frame.copy_(t.from_numpy(frame))
      self.reset_buf.copy_(t.from_numpy(reset))
      b['ctl'][:rows].copy_(t.from_numpy(ctl))
      ctx.set_input_shape('frame', tuple(self.frame.shape))
      ctx.set_input_shape('video_state', tuple(self.vs[0].shape))
      ctx.set_input_shape('video_reset', (1,))
      ctx.set_input_shape('point_state', (rows, L, P, C))
      ctx.set_input_shape('point_ctl', (rows, 4))
      addr = dict(frame=self.frame, video_state=self.vs[self.vping], video_reset=self.reset_buf,
                  point_state=b['ps'][p], point_ctl=b['ctl'], point_out=b['out'],
                  video_state_out=self.vs[1 - self.vping], point_state_out=b['ps'][1 - p])
      for name, tensor in addr.items():
        ctx.set_tensor_address(name, tensor.data_ptr())
      self.ev[0].record(self.stream)
      ctx.execute_async_v3(self.stream.cuda_stream)
      self.ev[1].record(self.stream)
      out = b['out'][:rows].cpu()
    self.stream.synchronize()
    self.gpu_ms = (0.0, self.ev[0].elapsed_time(self.ev[1]))
    self.ping, self.vping = 1 - p, 1 - self.vping
    return out.numpy()


def make_backend(kind, models_dir=None, tapnet_root=None, weights=None, onnx_path=None, engine=None,
                 resolution=DEFAULT_RES, new_points='prototype'):
  models_dir = models_dir or os.path.join(ROOT, 'models')
  if kind == 'torch':
    return TorchBackend(tapnet_root, weights, resolution, new_points=new_points)
  if kind in ('ort-cpu', 'ort-cuda'):
    onnx_path = onnx_path or os.path.join(models_dir, f'tapnextpp_step_{resolution}.onnx')
    return OrtBackend(onnx_path, device=kind.split('-')[1], new_points=new_points)
  if kind == 'trt':
    return TrtBackend(engine or os.path.join(models_dir, f'tapnextpp_step_{resolution}.engine'),
                      new_points=new_points)
  raise ValueError(kind)
