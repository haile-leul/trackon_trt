"""Export TAPNext++ (512x512 input) to a one-frame streaming ONNX graph and verify it.

  python export_tapnextpp_onnx.py --tapnet-root ../tapnet --weights ../tapnextpp_512.ckpt

writes  models/tapnextpp_step_512.onnx
          inputs   frame        [1,3,512,512]      RGB 0..255
                   video_state  [12,4,4096,768]    recurrent state of the 64x64 patch tokens
                   video_reset  [1]                1 = start a new sequence on this frame
                   point_state  [N,12,4,768]       recurrent state, one row per point
                   point_ctl    [N,4]              qx, qy (512 px), token (0 mask / 1 query / 2 unknown), reset
          outputs  point_out    [N,4]              x, y (512 px), visibility logit, certainty
                   video_state_out, point_state_out   next frame's state
        (N is dynamic; see tapnextpp_export/step.py for how points are added / removed)

Checkpoint: the TAPNext++ 512 checkpoint used by the official VOTS 2026 tracker,
  https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt
(Lightning checkpoint, 2.5 GB with optimizer state; only the weights are used).
Model code: git clone https://github.com/google-deepmind/tapnet

Verification (CPU, a few frames; ~20 s per frame at 512):
  1. TapNextPPStep (PyTorch) against the official TAPNext forward with its own
     online state (TAPNextTrackingState), with one point queried late;
  2. the ONNX graph in onnxruntime against TapNextPPStep, with points added
     (prototype copy and reset) and removed mid-clip.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tapnextpp_export import step as S  # pylint: disable=g-import-not-at-top

IN_NAMES = ['frame', 'video_state', 'video_reset', 'point_state', 'point_ctl']
OUT_NAMES = ['point_out', 'video_state_out', 'point_state_out']


def test_frames(n, res, video=None):
  """n uint8 RGB frames [res,res,3]: from a video if given, else a moving texture."""
  import cv2  # pylint: disable=g-import-not-at-top
  out = []
  if video:
    cap = cv2.VideoCapture(video)
    while len(out) < n:
      ok, f = cap.read()
      if not ok:
        break
      out.append(cv2.cvtColor(cv2.resize(f, (res, res), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    if len(out) == n:
      return out
  rng = np.random.default_rng(0)
  base = sum(cv2.resize(rng.random((res // s + 2, res // s + 2, 3)).astype(np.float32), (res + 96, res + 96),
                        interpolation=cv2.INTER_CUBIC) for s in (4, 8, 16, 32))
  base = ((base - base.min()) / (base.max() - base.min()) * 255).astype(np.uint8)
  return [np.ascontiguousarray(base[5 + 3 * t:5 + 3 * t + res, 10 + 5 * t:10 + 5 * t + res]) for t in range(n)]


def as_frame(rgb):
  return torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1)[None].contiguous()


def check_official(model, step, frames):
  """Official TAPNext (online state API) vs TapNextPPStep. Points A at frame 0, point B
  queried at frame 1 (official: `unknown` token on frame 0; ours: prototype row)."""
  r = step.resolution
  k = r / S.MODEL_SPACE
  a = np.array([[60, 50], [128, 128], [200, 90], [90, 210]], np.float32)  # model space (x, y)
  b = np.array([[160, 170]], np.float32)
  xy = np.concatenate([a, b])
  qt = np.array([0, 0, 0, 0, 1], np.float32)
  qp = torch.from_numpy(np.stack([qt, xy[:, 1], xy[:, 0]], 1))[None]
  state, worst = None, 0.0
  vs, ps = S.zero_state(step, 5)  # rows: A0..A3, prototype (becomes B on frame 1)
  for t, f in enumerate(frames):
    video = (torch.from_numpy(f).float() / 127.5 - 1)[None, None]
    with torch.no_grad():
      if state is None:
        tr, _, vis, state = model(video=video, query_points=qp)
      else:
        tr, _, vis, state = model(video=video, state=state)
      ctl = torch.zeros(5, 4)
      ctl[:, 0:2] = torch.from_numpy(xy * k)
      ctl[:4, 2] = S.TOK_QUERY if t == 0 else S.TOK_MASK
      ctl[4, 2] = S.TOK_UNKNOWN if t == 0 else (S.TOK_QUERY if t == 1 else S.TOK_MASK)
      out, vs, ps = step(as_frame(f), vs, torch.tensor([float(t == 0)]), ps, ctl)
    ref = tr[0, 0].numpy()[:, ::-1] * k
    act = qt <= t
    dxy = float(np.abs(out[:, :2].numpy() - ref)[act].max())
    dv = float(np.abs(out[:, 2].numpy() - vis[0, 0, :, 0].numpy())[act].max())
    worst = max(worst, dxy)
    print(f'  frame {t}: max|dxy| {dxy:.2e} px  max|dvis_logit| {dv:.2e}')
  return worst


class Rows:
  """Host-side row bookkeeping shared by the torch and onnxruntime runs of check_onnx:
  row 0 = prototype (unknown token), then the points."""

  def __init__(self, ps):
    self.ps = ps
    self.xy = np.zeros((1, 2), np.float32)

  def step_inputs(self, new_xy, reset_new, first):
    n0 = len(self.xy)
    k = len(new_xy)
    if k:
      src = np.zeros((k,) + self.ps.shape[1:], np.float32) if (reset_new or first) else \
          np.repeat(self.ps[:1], k, 0)
      self.ps = np.concatenate([self.ps, src])
      self.xy = np.concatenate([self.xy, new_xy])
    ctl = np.zeros((len(self.xy), 4), np.float32)
    ctl[:, 0:2] = self.xy
    ctl[0, 2] = S.TOK_UNKNOWN
    ctl[n0:, 2] = S.TOK_QUERY
    if reset_new:
      ctl[n0:, 3] = 1
    return ctl

  def remove(self, rows):
    keep = [i for i in range(len(self.xy)) if i - 1 not in set(rows)]
    self.ps, self.xy = self.ps[keep], self.xy[keep]


def check_onnx(onnx_path, step, frames, providers):
  import onnxruntime as ort  # pylint: disable=g-import-not-at-top
  so = ort.SessionOptions()
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  sess = ort.InferenceSession(onnx_path, so, providers=providers)
  r = step.resolution
  sched = {0: np.array([[80, 80], [300, 200], [450, 420]], np.float32) * r / 512,
           1: np.array([[200, 300], [400, 100]], np.float32) * r / 512,   # prototype copy
           2: np.array([[100, 400]], np.float32) * r / 512}               # reset
  vs0, ps0 = S.zero_state(step, 1)
  rows_t, rows_o = Rows(ps0.numpy()), Rows(ps0.numpy())
  vs_t, vs_o = vs0, vs0.numpy()
  worst, times = 0.0, []
  for t, f in enumerate(frames):
    if t == 3:
      rows_t.remove([1, 3])
      rows_o.remove([1, 3])
    new = sched.get(t, np.zeros((0, 2), np.float32))
    ctl_t = rows_t.step_inputs(new, reset_new=t == 2, first=t == 0)
    ctl_o = rows_o.step_inputs(new, reset_new=t == 2, first=t == 0)
    fr = as_frame(f)
    reset = np.array([float(t == 0)], np.float32)
    with torch.no_grad():
      out_t, vs_t, ps_t = step(fr, vs_t, torch.from_numpy(reset), torch.from_numpy(rows_t.ps),
                               torch.from_numpy(ctl_t))
    rows_t.ps = ps_t.numpy()
    t0 = time.perf_counter()
    out_o, vs_o, rows_o.ps = sess.run(None, dict(zip(IN_NAMES, [fr.numpy(), vs_o, reset, rows_o.ps, ctl_o])))
    times.append(time.perf_counter() - t0)
    dxy = float(np.abs(out_t[1:, :2].numpy() - out_o[1:, :2]).max())
    dv = float(np.abs(out_t[1:, 2].numpy() - out_o[1:, 2]).max())
    worst = max(worst, dxy)
    print(f'  frame {t}: points {len(ctl_o) - 1}  max|dxy| {dxy:.2e} px  max|dvis_logit| {dv:.2e}  '
          f'onnxruntime {times[-1]:.1f} s')
  return worst


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--tapnet-root', required=True, help='clone of github.com/google-deepmind/tapnet')
  ap.add_argument('--weights', required=True, help='tapnextpp_512.ckpt (or a plain state dict)')
  ap.add_argument('--resolution', type=int, default=512, help='input size the checkpoint was fine-tuned at')
  ap.add_argument('--out', help='default models/tapnextpp_step_<resolution>.onnx')
  ap.add_argument('--max-points', type=int, default=4096, help='upper bound of the dynamic N axis')
  ap.add_argument('--opset', type=int, default=18)
  ap.add_argument('--video', help='optional video for the verification clip')
  ap.add_argument('--verify-frames', type=int, default=4)
  ap.add_argument('--skip-export', action='store_true', help='only re-run the verification')
  ap.add_argument('--skip-verify', action='store_true')
  args = ap.parse_args()

  out = args.out or os.path.join(ROOT, 'models', f'tapnextpp_step_{args.resolution}.onnx')
  os.makedirs(os.path.dirname(out), exist_ok=True)
  model = S.load_tapnextpp(args.tapnet_root, args.weights)
  step = S.TapNextPPStep(model, args.resolution).eval()
  print(f'TAPNext++: input {args.resolution}x{args.resolution}, {step.num_video} patch tokens, '
        f'{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M weights')

  if not args.skip_export:
    # the 256x256x768 query table is replaced by the analytic embedding: do not store it
    query_table = model._buffers.pop('query_pos_embed')  # pylint: disable=protected-access
    n = 5
    vs, ps = S.zero_state(step, n)
    ctl = torch.zeros(n, 4)
    ctl[:, 0:2] = torch.rand(n, 2) * args.resolution
    ex = (torch.rand(1, 3, args.resolution, args.resolution) * 255, vs, torch.ones(1), ps, ctl)
    pn = torch.export.Dim('num_points', min=1, max=args.max_points)
    dyn = {'frame': None, 'video_state': None, 'video_reset': None,
           'point_state': {0: pn}, 'point_ctl': {0: pn}}
    print('exporting ...')
    with torch.no_grad():
      prog = torch.onnx.export(step, ex, input_names=IN_NAMES, output_names=OUT_NAMES, dynamic_shapes=dyn,
                               opset_version=args.opset, dynamo=True, optimize=True)
    prog.save(out)
    model.register_buffer('query_pos_embed', query_table)
    print('  saved', out, f'{os.path.getsize(out) / 1e6:.1f} MB')

  if args.skip_verify:
    return
  frames = test_frames(args.verify_frames, args.resolution, args.video)
  print('1) TapNextPPStep vs official TAPNext (online state API, one late query):')
  w1 = check_official(model, step, frames[:3])
  print('2) ONNX (onnxruntime CPU) vs TapNextPPStep (points added and removed):')
  w2 = check_onnx(out, step, frames, ['CPUExecutionProvider'])
  print(f'worst: official {w1:.2e} px, onnx {w2:.2e} px ({args.resolution} px coordinates)')
  print('OK' if max(w1, w2) < 0.05 else 'WARNING: deviation >= 0.05 px')


if __name__ == '__main__':
  main()
