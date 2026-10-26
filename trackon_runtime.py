"""Track-On2 runtime: one interface over PyTorch / ONNX Runtime / TensorRT, plus
a point manager that lets you add and remove points at any frame.

Per-point state = one row in (q_init [N,D], memory [N,M,D], memory_mask [N,M]),
exactly as in the official streaming Predictor. Adding a point appends a row
(is_new = 1 on that frame); removing a point deletes its row. There is no fixed
pool: N can be anything from 1 up to the engine's max (ONNX: 8192 by default).

Backends (all keep the state on the device that runs the model):
  torch     the original Track-On2 code (extract_frame_features + track_frame)
  ort-cpu   onnxruntime CPU
  ort-cuda  onnxruntime-gpu CUDA EP
  trt       TensorRT engines from tools/build_trt_engines.py (needs tensorrt + torch)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ENC_OUT = ['f4', 'f8', 'f16', 'f32', 'f_fused']
TRK_IN = ENC_OUT + ['q_init', 'memory', 'memory_mask', 'new_xy', 'is_new']
TRK_OUT = ['points', 'vis_logit', 'unc_logit', 'q_init_out', 'memory_out', 'memory_mask_out']
INPUT_H, INPUT_W = 384, 512


def import_tensorrt():
  """Imports tensorrt; TENSORRT_DIR (a TensorRT zip install) puts its DLLs first
  so Python uses exactly the TensorRT your C++ project links. Warns if the
  version differs from TENSORRT_VERSION."""
  trt_dir = os.environ.get('TENSORRT_DIR')
  if trt_dir:
    for sub in ('bin', 'lib'):
      d = os.path.join(trt_dir, sub)
      if os.path.isdir(d):
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
          os.add_dll_directory(d)
  try:
    import torch  # noqa: F401  (Windows: load torch's CUDA runtime first)  pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    pass
  import tensorrt as trt  # pylint: disable=g-import-not-at-top
  want = os.environ.get('TENSORRT_VERSION')
  if want and trt.__version__ != want:
    print(f'WARNING: tensorrt {trt.__version__} != TENSORRT_VERSION {want}: '
          'an engine only loads in the TensorRT version that built it')
  return trt


def load_cuda_dlls_for_ort(ort):
  """Windows: let onnxruntime-gpu use the CUDA/cuDNN DLLs shipped with torch."""
  try:
    import torch  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    pass
  if hasattr(ort, 'preload_dlls'):
    try:
      ort.preload_dlls()
    except Exception:  # pylint: disable=broad-except
      pass


class Backend:
  """Common state handling. Subclasses implement _encode, _track, and the
  device-side row operations _append(k) and _keep(idx)."""

  name = '?'
  d = 256
  m = 72
  gpu_ms = None  # optional: pure device time of the last step

  @property
  def n(self):
    return self._n

  def step(self, frame_rgb, new_xy):
    """frame_rgb: [384,512,3] uint8/float RGB (model input size).
    new_xy: [k,2] model-input pixel coords of points to add on this frame (k may be 0).
    Returns points [N,2] (model-input px), vis_prob [N], unc_logit [N]; the new
    points are the last k rows."""
    k = len(new_xy)
    if k:
      self._append(k)
    is_new = np.zeros(self._n, np.float32)
    xy = np.zeros((self._n, 2), np.float32)
    if k:
      is_new[-k:] = 1
      xy[-k:] = new_xy
    if self._n == 0:  # nothing to track: skip the model entirely
      return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    frame = np.ascontiguousarray(np.asarray(frame_rgb, np.float32).transpose(2, 0, 1)[None])
    feats = self._encode(frame)
    p, v, u = self._track(feats, xy, is_new)
    return p, 1.0 / (1.0 + np.exp(-v)), u

  def remove(self, rows):
    """Remove points by row index (rows of the last step's outputs)."""
    rows = set(int(r) for r in rows)
    if rows:
      self._keep(np.array([i for i in range(self._n) if i not in rows], np.int64))


class TorchBackend(Backend):
  """The original Track-On2 code path (reference)."""

  def __init__(self, trackon_root, ckpt, config=None, device=None, fp16=False, fake_dinov3=False):
    import torch  # pylint: disable=g-import-not-at-top
    sys.path.insert(0, ROOT)
    from trackon_export.loading import build_model  # pylint: disable=g-import-not-at-top
    self.torch = torch
    self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    self.model, _ = build_model(trackon_root, ckpt, config, fake_dinov3=fake_dinov3)
    self.model.to(self.dev)
    self.fp16 = fp16 and self.dev.type == 'cuda'
    self.d, self.m = self.model.D, self.model.M
    self.name = f'torch {torch.__version__} {self.dev.type}{" fp16" if self.fp16 else ""}'
    self._n = 0
    z = lambda *s: torch.zeros(*s, device=self.dev)
    self.q, self.mem = z(0, self.d), z(0, self.m, self.d)
    self.mask = torch.ones(0, self.m, dtype=torch.bool, device=self.dev)

  def _append(self, k):
    t = self.torch
    self.q = t.cat([self.q, t.zeros(k, self.d, device=self.dev)])
    self.mem = t.cat([self.mem, t.zeros(k, self.m, self.d, device=self.dev)])
    self.mask = t.cat([self.mask, t.ones(k, self.m, dtype=t.bool, device=self.dev)])
    self._n += k

  def _keep(self, idx):
    i = self.torch.as_tensor(idx, device=self.dev)
    self.q, self.mem, self.mask = self.q[i], self.mem[i], self.mask[i]
    self._n = len(idx)

  def _encode(self, frame):
    t = self.torch
    with t.no_grad(), t.autocast('cuda', dtype=t.float16, enabled=self.fp16):
      return self.model.extract_frame_features(t.from_numpy(frame).to(self.dev))

  def _track(self, feats, xy, is_new):
    t, mdl = self.torch, self.model
    new = np.flatnonzero(is_new)
    with t.no_grad(), t.autocast('cuda', dtype=t.float16, enabled=self.fp16):
      if len(new):  # Predictor.init_queries
        fmap = feats[4].view(1, mdl.Hf, mdl.Wf, mdl.D).permute(0, 3, 1, 2).float()
        g = t.from_numpy(xy[new] / np.array([mdl.W, mdl.H], np.float32) * 2 - 1).to(self.dev)
        qn = t.nn.functional.grid_sample(fmap, g.view(1, -1, 1, 2), mode='bilinear',
                                         padding_mode='border', align_corners=False)[0, :, :, 0].T
        self.q[t.as_tensor(new, device=self.dev)] = qn.float()
      p, v, qnew = mdl.track_frame(self.q, self.mask, self.mem, feats, mdl.H, mdl.W)
    self.mem = t.roll(self.mem, -1, 1)
    self.mem[:, -1] = qnew.float()
    self.mask = t.roll(self.mask, -1, 1)
    self.mask[:, -1] = False
    if self.dev.type == 'cuda':
      t.cuda.synchronize()
    return p.float().cpu().numpy(), v.float().cpu().numpy(), np.zeros(self._n, np.float32)


class OrtBackend(Backend):
  """onnxruntime; per-point state stays on the device between frames and only
  round-trips to the host when points are added or removed."""

  def __init__(self, models_dir, device='cpu'):
    import onnxruntime as ort  # pylint: disable=g-import-not-at-top
    if device == 'cuda':
      load_cuda_dlls_for_ort(ort)
      providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
      providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.enc = ort.InferenceSession(os.path.join(models_dir, 'trackon_encoder.onnx'), so, providers=providers)
    self.trk = ort.InferenceSession(os.path.join(models_dir, 'trackon_tracker.onnx'), so, providers=providers)
    if device == 'cuda' and self.enc.get_providers()[0] != 'CUDAExecutionProvider':
      raise RuntimeError('CUDA EP not usable (run tools/check_env.py)')
    self.ort, self.device = ort, device
    shp = {i.name: i.shape for i in self.trk.get_inputs()}
    self.d, self.m = shp['q_init'][1], shp['memory'][1]
    self.name = f'ort-{device} ({self.enc.get_providers()[0]})'
    self._n = 0
    self.state = None  # (q, mem, mask) OrtValues on device

  def _host_state(self):
    if self.state is None:
      return (np.zeros((0, self.d), np.float32), np.zeros((0, self.m, self.d), np.float32),
              np.zeros((0, self.m), np.float32))
    return tuple(s.numpy() for s in self.state)

  def _set_state(self, q, mem, mask):
    up = lambda a: self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(a, np.float32), self.device, 0)
    self.state = (up(q), up(mem), up(mask)) if len(q) else None

  def _append(self, k):
    q, mem, mask = self._host_state()
    self._set_state(np.concatenate([q, np.zeros((k, self.d), np.float32)]),
                    np.concatenate([mem, np.zeros((k, self.m, self.d), np.float32)]),
                    np.concatenate([mask, np.ones((k, self.m), np.float32)]))
    self._n += k

  def _keep(self, idx):
    q, mem, mask = self._host_state()
    self._set_state(q[idx], mem[idx], mask[idx])
    self._n = len(idx)

  def _encode(self, frame):
    io = self.enc.io_binding()
    io.bind_cpu_input('frame', frame)
    for name in ENC_OUT:
      io.bind_output(name, self.device)
    self.enc.run_with_iobinding(io)
    return io.get_outputs()

  def _track(self, feats, xy, is_new):
    io = self.trk.io_binding()
    for name, v in zip(ENC_OUT, feats):
      io.bind_ortvalue_input(name, v)
    for name, v in zip(('q_init', 'memory', 'memory_mask'), self.state):
      io.bind_ortvalue_input(name, v)
    io.bind_cpu_input('new_xy', xy)
    io.bind_cpu_input('is_new', is_new)
    for name in TRK_OUT[:3]:
      io.bind_output(name, 'cpu')
    for name in TRK_OUT[3:]:
      io.bind_output(name, self.device)
    self.trk.run_with_iobinding(io)
    out = io.get_outputs()
    self.state = tuple(out[3:])
    return out[0].numpy(), out[1].numpy(), out[2].numpy()


class TrtBackend(Backend):
  """TensorRT engines; all buffers on the GPU (torch tensors), capacity grows x2."""

  def __init__(self, encoder_engine, tracker_engine):
    trt = import_tensorrt()
    import torch  # pylint: disable=g-import-not-at-top
    self.trt, self.torch = trt, torch
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    load = lambda p: rt.deserialize_cuda_engine(open(p, 'rb').read())
    self.enc_engine, self.trk_engine = load(encoder_engine), load(tracker_engine)
    if self.enc_engine is None or self.trk_engine is None:
      raise RuntimeError('could not load engines (built with another TensorRT version / GPU?)')
    self.enc_ctx = self.enc_engine.create_execution_context()
    self.trk_ctx = self.trk_engine.create_execution_context()
    self.stream = torch.cuda.Stream()
    self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    shp = lambda e, n: tuple(e.get_tensor_shape(n))
    self.d, self.m = shp(self.trk_engine, 'q_init')[1], shp(self.trk_engine, 'memory')[1]
    prof = self.trk_engine.get_tensor_profile_shape('q_init', 0)
    self.min_n, self.max_n = prof[0][0], prof[2][0]
    self.name = f'trt {trt.__version__} (points {self.min_n}..{self.max_n})'
    cuda = lambda s: torch.zeros(s, dtype=torch.float32, device='cuda')
    self.frame = cuda(shp(self.enc_engine, 'frame'))
    self.feats = [cuda(shp(self.enc_engine, n)) for n in ENC_OUT]
    self.enc_ctx.set_input_shape('frame', tuple(self.frame.shape))
    for name, t in [('frame', self.frame)] + list(zip(ENC_OUT, self.feats)):
      self.enc_ctx.set_tensor_address(name, t.data_ptr())
    for name, t in zip(ENC_OUT, self.feats):
      self.trk_ctx.set_input_shape(name, tuple(t.shape))
      self.trk_ctx.set_tensor_address(name, t.data_ptr())
    self._n, self.cap = 0, 0
    self._alloc(64)

  def _alloc(self, cap):
    """(Re)allocate per-point buffers for `cap` rows, keeping the first n rows."""
    t, d, m, n = self.torch, self.d, self.m, self._n
    cuda = lambda *s: t.zeros(*s, dtype=t.float32, device='cuda')
    new = dict(q=[cuda(cap, d), cuda(cap, d)], mem=[cuda(cap, m, d), cuda(cap, m, d)],
               mask=[cuda(cap, m), cuda(cap, m)], xy=cuda(cap, 2), is_new=cuda(cap),
               points=cuda(cap, 2), vis=cuda(cap), unc=cuda(cap))
    if self.cap:
      for key in ('q', 'mem', 'mask'):
        new[key][0][:n] = self.buf[key][self.ping][:n]
    self.buf, self.cap, self.ping = new, cap, 0

  def _append(self, k):
    if self._n + k > self.max_n:
      raise RuntimeError(f'{self._n + k} points > engine max {self.max_n}: rebuild with a larger --max-points')
    if self._n + k > self.cap:
      self._alloc(max(2 * self.cap, self._n + k))
    b, p, n = self.buf, self.ping, self._n
    b['q'][p][n:n + k] = 0
    b['mem'][p][n:n + k] = 0
    b['mask'][p][n:n + k] = 1
    self._n += k

  def _keep(self, idx):
    i = self.torch.as_tensor(idx, device='cuda')
    b, p = self.buf, self.ping
    for key in ('q', 'mem', 'mask'):
      b[key][p][:len(idx)] = b[key][p][i]
    self._n = len(idx)

  def _encode(self, frame):
    with self.torch.cuda.stream(self.stream):
      self.frame.copy_(self.torch.from_numpy(frame), non_blocking=False)
      self.ev[0].record(self.stream)
      self.enc_ctx.execute_async_v3(self.stream.cuda_stream)
      self.ev[1].record(self.stream)
    return self.feats

  def _track(self, feats, xy, is_new):
    t, b, n, p = self.torch, self.buf, self._n, self.ping
    if n < self.min_n:
      raise RuntimeError(f'engine needs >= {self.min_n} points')
    ctx = self.trk_ctx
    with t.cuda.stream(self.stream):
      b['xy'][:n].copy_(t.from_numpy(xy))
      b['is_new'][:n].copy_(t.from_numpy(is_new))
      shapes = dict(q_init=(n, self.d), memory=(n, self.m, self.d), memory_mask=(n, self.m),
                    new_xy=(n, 2), is_new=(n,))
      for name, s in shapes.items():
        ctx.set_input_shape(name, s)
      addr = dict(q_init=b['q'][p], memory=b['mem'][p], memory_mask=b['mask'][p], new_xy=b['xy'],
                  is_new=b['is_new'], points=b['points'], vis_logit=b['vis'], unc_logit=b['unc'],
                  q_init_out=b['q'][1 - p], memory_out=b['mem'][1 - p], memory_mask_out=b['mask'][1 - p])
      for name, tensor in addr.items():
        ctx.set_tensor_address(name, tensor.data_ptr())
      ctx.execute_async_v3(self.stream.cuda_stream)
      self.ev[2].record(self.stream)
      out = [b['points'][:n].cpu(), b['vis'][:n].cpu(), b['unc'][:n].cpu()]
    self.stream.synchronize()
    self.gpu_ms = (self.ev[0].elapsed_time(self.ev[1]), self.ev[1].elapsed_time(self.ev[2]))
    self.ping = 1 - p
    return tuple(o.numpy() for o in out)


def make_backend(kind, models_dir=None, trackon_root=None, ckpt=None, config=None,
                 encoder_engine=None, tracker_engine=None, fp16=False, fake_dinov3=False):
  if kind == 'torch':
    return TorchBackend(trackon_root, ckpt, config, fp16=fp16, fake_dinov3=fake_dinov3)
  if kind in ('ort-cpu', 'ort-cuda'):
    return OrtBackend(models_dir, device=kind.split('-')[1])
  if kind == 'trt':
    models_dir = models_dir or os.path.join(ROOT, 'models')
    return TrtBackend(encoder_engine or os.path.join(models_dir, 'trackon_encoder.engine'),
                      tracker_engine or os.path.join(models_dir, 'trackon_tracker.engine'))
  raise ValueError(kind)


class PointTracker:
  """Points with stable ids in ORIGINAL image pixels on top of a Backend.

  add(x, y)    -> id, starts tracking on the next step()
  remove(ids)  stops tracking those ids (takes effect immediately)
  step(frame_bgr) -> dict id -> (x, y, vx, vy, vis_prob, visible)
  The frame (or `roi` of it) is resized to the model input (Track-On2 512x384,
  TAPNext++ 512x512), exactly like the official code resizes the full frame.
  Works with the Track-On2 backends here and the TAPNext++ ones in tapnextpp_runtime.
  """

  def __init__(self, backend, roi=None, delta_v=None):
    self.b, self.roi = backend, roi
    self.delta_v = delta_v if delta_v is not None else getattr(backend, 'delta_v', 0.8)
    self.in_w, self.in_h = getattr(backend, 'input_size', (INPUT_W, INPUT_H))
    self.ids = []            # row -> id
    self.pending = []        # (id, x, y) to add on the next step
    self.state = {}          # id -> dict
    self.next_id = 0
    self.t = 0

  def _scale(self, w, h):
    rx, ry, rw, rh = self.roi if self.roi else (0, 0, w, h)
    return rx, ry, rw / self.in_w, rh / self.in_h

  def add(self, x, y):
    i = self.next_id
    self.next_id += 1
    self.pending.append((i, float(x), float(y)))
    return i

  def remove(self, ids):
    ids = set(ids)
    self.pending = [p for p in self.pending if p[0] not in ids]
    rows = [r for r, i in enumerate(self.ids) if i in ids]
    self.b.remove(rows)
    self.ids = [i for i in self.ids if i not in ids]
    for i in ids:
      self.state.pop(i, None)

  @property
  def num_points(self):
    return len(self.ids) + len(self.pending)

  def median_velocity(self, ids, require_visible=True):
    """Median (vx, vy) in px/frame of the given ids (points that already have a
    velocity and, if require_visible, are visible), plus the number of points
    used; (None, None, 0) if none."""
    v = [(s['vx'], s['vy']) for i in ids for s in [self.state.get(i)]
         if s and (s['visible'] or not require_visible) and s['has_v']]
    if not v:
      return None, None, 0
    v = np.array(v)
    return float(np.median(v[:, 0])), float(np.median(v[:, 1])), len(v)

  def step(self, frame_bgr):
    import cv2  # pylint: disable=g-import-not-at-top
    h, w = frame_bgr.shape[:2]
    rx, ry, sx, sy = self._scale(w, h)
    crop = frame_bgr if not self.roi else frame_bgr[int(ry):int(ry + sy * self.in_h), int(rx):int(rx + sx * self.in_w)]
    rgb = cv2.cvtColor(cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    new = np.array([[(x - rx) / sx, (y - ry) / sy] for _, x, y in self.pending], np.float32).reshape(-1, 2)
    self.ids += [p[0] for p in self.pending]
    born = {p[0] for p in self.pending}
    self.pending = []
    pts, vis, _ = self.b.step(rgb, new)
    out = {}
    for row, i in enumerate(self.ids):
      x, y = rx + pts[row, 0] * sx, ry + pts[row, 1] * sy
      prev = self.state.get(i)
      vx, vy = (0.0, 0.0) if (prev is None or i in born) else (x - prev['x'], y - prev['y'])
      lost = 0 if vis[row] >= self.delta_v else (prev['lost'] + 1 if prev else 1)
      s = dict(x=x, y=y, vx=vx, vy=vy, vis=float(vis[row]), visible=bool(vis[row] >= self.delta_v), lost=lost,
               has_v=not (prev is None or i in born))
      self.state[i] = s
      out[i] = s
    self.t += 1
    return out
