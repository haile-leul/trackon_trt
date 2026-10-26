"""Build the TAPNext++ TensorRT engine from models/tapnextpp_step_512.onnx and verify it.

  python tools/build_tapnextpp_engine.py                     # mixed precision, points 0..255
  python tools/build_tapnextpp_engine.py --max-points 1023 --opt-points 64
  python tools/build_tapnextpp_engine.py --verify-only

Writes models/tapnextpp_step_512.engine (same tensor names as the ONNX; run it with
run_video.py --model tapnextpp --backend trt). Engines only load in the TensorRT
version and GPU architecture that built them.

The engine's point axis counts ROWS = points + 1 (row 0 is the prototype row
the runtime keeps for adding points), so --max-points P gives a profile of
1..P+1 rows.

Precision (--precision):
  mixed  fp16 with fp32 islands: LayerNorms, RMSNorm reductions, the RG-LRU
         decay / state update (a ~ 1 - 1e-4 rounds to 1 in fp16), everything
         that reads the recurrent state, and the output heads (coordinates).
         The heavy parts (patch embedding, all Linear layers, attention, MLPs)
         run in fp16. Needs a GPU with fast fp16 (tensor cores: RTX 20xx and
         newer). The state inputs / outputs stay fp32.
  fp16   TensorRT decides everything (fastest, may drift).
  fp32   reference (use this on GPUs without fast fp16, e.g. GTX 16xx).

Cost: ~2.1 TFLOP per frame at 512x512 (4096 patch tokens through 12 layers of
SSM + full attention), plus 1.2 GB of recurrent video state read and written
per frame. GPU memory: ~0.4-0.8 GB weights + 2 x 0.6 GB state + activations.
"""

import argparse
import collections
import os
import sys

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tapnextpp_runtime import L, C, P, import_tensorrt, make_backend  # pylint: disable=g-import-not-at-top

TAG = 'fp32__'
ELEMENTWISE = {'Mul', 'Add', 'Sub', 'Div', 'Sqrt', 'Reciprocal', 'Pow', 'Where', 'Exp', 'Neg', 'Max',
               'Min', 'Sigmoid', 'Softplus', 'Log', 'Clip'}
SHAPE_OPS = {'Gather', 'Slice', 'Concat', 'Unsqueeze', 'Squeeze', 'Reshape', 'Transpose', 'Split'}


def tag_fp32_nodes(model):
  """Prefix the names of ONNX nodes that must stay fp32 with TAG. Returns count."""
  g = model.graph
  cons = collections.defaultdict(list)
  prod = {}
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
    for o in n.output:
      prod[o] = n
  tagged = set()

  def forward(starts, hops, ops):
    frontier = list(starts)
    for _ in range(hops):
      nxt = []
      for n in frontier:
        for o in n.output:
          for c in cons[o]:
            if c.op_type in ops and id(c) not in tagged:
              tagged.add(id(c))
              nxt.append(c)
      frontier = nxt

  for n in g.node:
    if n.op_type == 'LayerNormalization':
      tagged.add(id(n))
    elif n.op_type == 'ReduceMean':  # RMSNorm: (x s)^2 -> mean -> + eps -> rsqrt
      tagged.add(id(n))
      p = prod.get(n.input[0])
      if p is not None and p.op_type in ('Pow', 'Mul'):
        tagged.add(id(p))
      forward([n], 3, {'Add', 'Sqrt', 'Reciprocal', 'Div'})
    elif n.op_type == 'Exp':  # RG-LRU: a, a^2, sqrt(1 - a^2), a * h + x
      tagged.add(id(n))
      forward([n], 6, ELEMENTWISE)
    elif n.op_type == 'Softplus':
      tagged.add(id(n))

  # everything that reads the recurrent state (keep-mask, conv sum, a * h)
  starts = [c for name in ('video_state', 'point_state', 'video_reset') for c in cons[name]]
  for c in starts:
    tagged.add(id(c))
  forward(starts, 6, ELEMENTWISE | SHAPE_OPS)

  # output heads: all descendants of the encoder norm (= the last LayerNorm that
  # still has the 4 LayerNorms of the two heads below it)
  def descendants(start):
    stack, seen = [start], {}
    while stack:
      n = stack.pop()
      if id(n) in seen:
        continue
      seen[id(n)] = n
      for o in n.output:
        stack.extend(cons[o])
    return seen

  lns = [n for n in g.node if n.op_type == 'LayerNormalization']
  for ln in reversed(lns):
    desc = descendants(ln)
    if sum(1 for d in desc.values() if d.op_type == 'LayerNormalization') >= 5:  # itself + 4
      tagged.update(desc.keys())
      break

  count = 0
  for n in g.node:
    if id(n) in tagged:
      n.name = TAG + (n.name or f'{n.op_type}_{count}')
      count += 1
  return count


def build(trt, onnx_path, engine_path, precision, rows, workspace_gb, verbose):
  logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
  model = onnx.load(onnx_path)
  if precision == 'mixed':
    print(f'  tagged {tag_fp32_nodes(model)} ONNX nodes to stay fp32')
  res = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  v = (res // 8) ** 2
  builder = trt.Builder(logger)
  flags = 0
  if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  network = builder.create_network(flags)
  parser = trt.OnnxParser(network, logger)
  if not parser.parse(model.SerializeToString()):
    for i in range(parser.num_errors):
      print(parser.get_error(i))
    sys.exit(f'ONNX parse failed: {onnx_path}')
  del model
  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
  if precision in ('mixed', 'fp16'):
    config.set_flag(trt.BuilderFlag.FP16)
  if precision == 'mixed':
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    floats = (trt.float32, trt.float16)
    pinned = 0
    import warnings  # pylint: disable=g-import-not-at-top
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', DeprecationWarning)
      for i in range(network.num_layers):
        layer = network.get_layer(i)
        ident = layer.name + ' ' + (getattr(layer, 'metadata', '') or '')
        if TAG not in ident or layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
          continue
        if all(layer.get_output(k).dtype not in floats for k in range(layer.num_outputs)):
          continue
        layer.precision = trt.float32
        for k in range(layer.num_outputs):
          if layer.get_output(k).dtype in floats:
            layer.set_output_type(k, trt.float32)
        pinned += 1
    print(f'  pinned {pinned}/{network.num_layers} layers to fp32')
  prof = builder.create_optimization_profile()
  fixed = {'frame': (1, 3, res, res), 'video_state': (L, P, v, C), 'video_reset': (1,)}
  for name, s in fixed.items():
    prof.set_shape(name, s, s, s)
  prof.set_shape('point_state', *[(r, L, P, C) for r in rows])
  prof.set_shape('point_ctl', *[(r, 4) for r in rows])
  config.add_optimization_profile(prof)
  print(f'  building {os.path.basename(engine_path)} ({precision}, rows {rows[0]}..{rows[2]}) '
        '... this takes a while')
  blob = builder.build_serialized_network(network, config)
  if blob is None:
    sys.exit('engine build failed (see TensorRT messages above)')
  with open(engine_path, 'wb') as f:
    f.write(blob)
  print(f'  saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)')


def verify(onnx_path, engine_path, video=None, frames=8):
  from export_tapnextpp_onnx import test_frames  # pylint: disable=g-import-not-at-top
  trt_b = make_backend('trt', engine=engine_path)
  try:
    ref_b = make_backend('ort-cuda', onnx_path=onnx_path)
  except Exception as e:  # pylint: disable=broad-except
    print(f'  (onnxruntime CUDA unavailable: {e}; using CPU reference, slow)')
    ref_b = make_backend('ort-cpu', onnx_path=onnx_path)
  print(f'verifying {trt_b.name} against {ref_b.name}')
  res = trt_b.res
  clip = test_frames(frames, res, video)
  rng = np.random.default_rng(0)
  rand = lambda k: rng.uniform(0.05 * res, 0.95 * res, (k, 2)).astype(np.float32)
  sched = {0: rand(24), 3: rand(16)}
  worst, nan_trt, nan_ref, times = 0.0, 0, 0, []
  for t, f in enumerate(clip):
    if t == 5:  # remove every third point
      for b in (trt_b, ref_b):
        b.remove(range(0, b.n, 3))
    new = sched.get(t, np.zeros((0, 2), np.float32))
    pa, va, _ = trt_b.step(f, new)
    pb, vb, _ = ref_b.step(f, new)
    nan_trt += int(np.isnan(pa).sum() + np.isnan(va).sum())
    nan_ref += int(np.isnan(pb).sum() + np.isnan(vb).sum())
    dxy = float(np.nanmax(np.abs(pa - pb)))
    dv = float(np.nanmax(np.abs(va - vb)))
    worst = max(worst, dxy)
    times.append(trt_b.gpu_ms[1])
    print(f'  frame {t}: N={trt_b.n:3d}  max|dxy|={dxy:.3f}px  max|dvis|={dv:.4f}  '
          f'NaNs engine {nan_trt} ref {nan_ref}  GPU {trt_b.gpu_ms[1]:.1f} ms')
  if nan_trt or nan_ref:
    print('FAILED: NaN outputs. Rebuild with --precision fp32 to tell a precision problem from a setup problem.')
    sys.exit(1)
  print(f'WORST position difference {worst:.3f}px ({res}x{res} model pixels)')
  print(f'median GPU time {np.median(times[2:]):.1f} ms per frame (N~30)')
  print('OK' if worst < 1.0 else 'WARNING: engine deviates by >= 1 px from onnxruntime')


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--onnx', default=os.path.join(ROOT, 'models', 'tapnextpp_step_512.onnx'))
  ap.add_argument('--engine', help='default: the ONNX path with .engine')
  ap.add_argument('--precision', choices=['mixed', 'fp16', 'fp32'], default='mixed')
  ap.add_argument('--opt-points', type=int, default=32)
  ap.add_argument('--max-points', type=int, default=255)
  ap.add_argument('--workspace-gb', type=float, default=4)
  ap.add_argument('--video', help='video for the verification clip (default: synthetic)')
  ap.add_argument('--verify-only', action='store_true')
  ap.add_argument('--no-verify', action='store_true')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()
  engine = args.engine or os.path.splitext(args.onnx)[0] + '.engine'
  if not args.verify_only:
    trt = import_tensorrt()
    print('TensorRT', trt.__version__)
    build(trt, args.onnx, engine, args.precision, (1, args.opt_points + 1, args.max_points + 1),
          args.workspace_gb, args.verbose)
  if not args.no_verify:
    verify(args.onnx, engine, args.video)


if __name__ == '__main__':
  main()
