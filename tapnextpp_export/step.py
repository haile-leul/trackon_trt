"""TAPNext++ as a one-frame step with explicit recurrent state (export friendly).

TAPNext++ (google-deepmind/tapnet, tapnet/tapnext + tapnextpp) is a TRecViT:
every token goes through 12 blocks of [Griffin SSM over time -> ViT attention
over all tokens of the frame]. The tokens of a frame are the 64x64 image
patches (512x512 input, patch 8) and one token per tracked point. The only
memory between frames is the SSM state of every token in every layer:
  conv1d history (3 previous inputs) + RG-LRU hidden state = 4 x 768 floats.

So one frame is a pure function
  (frame, video_state, point rows) -> (predictions, video_state', point rows')
and points are independent rows: adding a point appends a row, removing one
deletes its row; the model is never re-initialised.

State planes per token and layer (PLANES = 4): conv history t-3, t-2, t-1, then
the RG-LRU state h.
  video_state  [L, 4, V, C]   V = (R/8)^2 patch tokens (4096 at R = 512)
  point_state  [N, L, 4, C]   one contiguous row per point
  point_ctl    [N, 4]         qx, qy (input pixels, R x R), token, reset
      token  0 = mask    (tracked point, the normal case)
             1 = query   (the point is defined at (qx, qy) on THIS frame)
             2 = unknown (not queried yet: the state a later query starts from)
      reset  1 = start this row from an empty state on this frame
  point_out    [N, 4]         x, y (input pixels), visibility logit, certainty

How a point is added online (same as the official model with a late query):
the official tracker feeds a point the `unknown` token on every frame before
its query frame, the query token on the query frame and the `mask` token after.
The runtime keeps one extra `unknown` row (the prototype) on every frame; a new
point starts from a copy of the prototype's state with token = query. With
reset = 1 instead, the point starts from an empty state (like frame 0).

Everything is written with plain ops: no einsum, the sin-cos query embedding is
computed analytically (no 256x256x768 table), GELU / RMSNorm are fp16-safe and
the per-token SSM runs separately on the video tokens and on the point tokens
(no concatenation of the 600 MB state; only the ViT attention mixes them).
"""

import math
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

NUM_LAYERS = 12
WIDTH = 768
PLANES = 4          # conv history (3) + RG-LRU state (1)
PATCH = 8
MODEL_SPACE = 256   # TAPNext coordinates / query embedding live in a 256 x 256 space
CTL = 4             # qx, qy, token, reset
OUT = 4             # x, y, visibility logit, certainty
TOK_MASK, TOK_QUERY, TOK_UNKNOWN = 0.0, 1.0, 2.0


def _gelu_tanh(x):
  """tanh-GELU (as the official JAX/torch code). The cube's input is clipped to
  +-20: there tanh(.) is already exactly +-1, so the result is unchanged, but
  x^3 can no longer overflow fp16."""
  xc = torch.clamp(x, -20.0, 20.0)
  return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * xc * xc * xc)))


def _rms_norm(mod, x, prescale=1.0 / 256):
  """x / sqrt(mean(x^2) + eps) * (scale + 1), computed on x * prescale (same
  value; x^2 of the residual stream (|x| ~ 6e3) would overflow fp16)."""
  xs = x * prescale
  var = torch.mean(xs * xs, dim=-1, keepdim=True)
  return xs * torch.rsqrt(var + mod.eps * prescale * prescale) * (mod.scale + 1)


def _block_diag_linear(mod, x):
  # x [M, C] -> [H, M, C/H] @ [H, C/H, C/H] -> [M, C]
  m = x.shape[0]
  h, bw, _ = mod.w.shape
  y = torch.matmul(x.reshape(m, h, bw).transpose(0, 1), mod.w) + mod.b.unsqueeze(1)
  return y.transpose(0, 1).reshape(m, h * bw)


def _sincos_at(xi, yi, c):
  """Row of posemb_sincos_2d(256, 256, c) at integer pixel (yi, xi): [Q,1] -> [Q,C]."""
  d = c // 4
  omega = torch.arange(d, dtype=torch.float32, device=xi.device) / (d - 1)
  omega = 1.0 / (10_000.0**omega)
  ax = xi * omega.view(1, d)
  ay = yi * omega.view(1, d)
  return torch.cat([torch.sin(ax), torch.cos(ax), torch.sin(ay), torch.cos(ay)], -1)


def query_posemb(qx, qy, c, size=MODEL_SPACE):
  """grid_sample(query_pos_embed, align_corners=False, zero padding) at (qx, qy)
  in model space, as bilinear interpolation of the analytic sin-cos table."""
  px, py = qx - 0.5, qy - 0.5
  x0, y0 = torch.floor(px), torch.floor(py)
  wx1, wy1 = px - x0, py - y0
  out = 0.0
  for xi, wx in ((x0, 1 - wx1), (x0 + 1, wx1)):
    for yi, wy in ((y0, 1 - wy1), (y0 + 1, wy1)):
      valid = ((xi >= 0) & (xi <= size - 1) & (yi >= 0) & (yi <= size - 1)).float()
      out = out + _sincos_at(xi, yi, c) * (wx * wy * valid)
  return out


class TapNextPPStep(nn.Module):
  """One TAPNext++ frame. See the module docstring for the tensors."""

  def __init__(self, model, resolution=512):
    super().__init__()
    if resolution % PATCH:
      raise ValueError(f'resolution {resolution} must be a multiple of {PATCH}')
    self.m = model
    self.resolution = resolution
    self.grid = resolution // PATCH
    self.num_video = self.grid * self.grid
    with torch.no_grad():  # learned 32x32 position embedding, bicubic to the patch grid
      pos = model._video_pos_emb(self.grid, self.grid)  # pylint: disable=protected-access
    self.register_buffer('video_pos', pos.reshape(self.num_video, WIDTH).float().clone())

  def _ssm(self, blk, x, st, keep):
    """Griffin residual block, one timestep. x [M, C]; st [4, M, C]; keep [M or 1, 1]."""
    c0, c1, c2, h = st[0] * keep, st[1] * keep, st[2] * keep, st[3] * keep
    raw = x
    xn = _rms_norm(blk.temporal_pre_norm, raw)
    rb = blk.recurrent_block
    y = _gelu_tanh(rb.linear_y(xn))
    xb = rb.linear_x(xn)
    w = rb.conv_1d.w  # [4, C]: weights for t-3 .. t
    xc = c0 * w[0] + c1 * w[1] + c2 * w[2] + xb * w[3] + rb.conv_1d.b
    lru = rb.rg_lru
    gate_x = torch.sigmoid(_block_diag_linear(lru.input_gate, xc))
    gate_a = torch.sigmoid(_block_diag_linear(lru.a_gate, xc))
    log_a = -8.0 * gate_a * F.softplus(lru.a_param)
    a = torch.exp(log_a)
    mult = torch.sqrt(1.0 - torch.exp(2.0 * log_a))
    mult = (1.0 - keep) + keep * mult  # first step of a token: no input normalisation
    nh = a * h + xc * gate_x * mult
    res = rb.linear_out(nh * y) + raw
    xm = _rms_norm(blk.channel_pre_norm, res)
    mlp = blk.mlp_block
    up = torch.matmul(xm, mlp.ffw_up.w) + mlp.ffw_up.b[:, 0]  # [2, M, 4C]
    out = mlp.ffw_down(_gelu_tanh(up[0]) * up[1]) + res
    return out, torch.stack([c1, c2, xb, nh], 0)

  def _point_tokens(self, ctl):
    m, c = self.m, WIDTH
    s = MODEL_SPACE / self.resolution
    pe = query_posemb(ctl[:, 0:1] * s, ctl[:, 1:2] * s, c)
    query = m.point_query_token.view(1, c) + pe
    tok = ctl[:, 2:3]
    t = torch.where((tok > 0.5) & (tok < 1.5), query, m.mask_token.view(1, c))
    return torch.where(tok > 1.5, m.unknown_token.view(1, c), t)

  def _heads(self, pts):
    m = self.m
    logits = m.coordinate_head(pts)  # [N, 512] = y logits | x logits (256 each)
    ly, lx = logits.chunk(2, dim=-1)
    n = MODEL_SPACE
    index = torch.arange(n, dtype=torch.float32, device=pts.device).view(1, n)

    def soft_argmax(lg):  # official: softmax(T=0.5) inside +-20 bins of the argmax, + 0.5
      am = lg.argmax(dim=-1, keepdim=True).float()
      p = F.softmax(lg * 0.5, dim=-1) * (torch.abs(am - index) <= 20).float()
      return (p * index).sum(-1, keepdim=True) / p.sum(-1, keepdim=True) + 0.5

    ty, tx = soft_argmax(ly), soft_argmax(lx)
    # certainty: probability mass within 8 px (model space) of the prediction
    centers = index + 0.5
    dy2 = (centers - ty) ** 2
    dx2 = (centers - tx) ** 2
    inside = ((dy2.unsqueeze(2) + dx2.unsqueeze(1)) <= 64.0 + 1e-8).float()
    cert = (F.softmax(ly, -1).unsqueeze(2) * F.softmax(lx, -1).unsqueeze(1) * inside).sum((1, 2))
    vis = m.visible_head(pts)[:, 0]
    k = self.resolution / MODEL_SPACE
    return torch.stack([tx[:, 0] * k, ty[:, 0] * k, vis, cert], -1)

  @staticmethod
  def _vit_points_only(vit, vt, pt):
    """torchvision EncoderBlock for the point tokens only (keys / values: all tokens)."""
    kv = vit.ln_1(torch.cat([vt, pt], 0).unsqueeze(0))
    q = kv[:, vt.shape[0]:]
    a, _ = vit.self_attention(q, kv, kv, need_weights=False)
    x = a[0] + pt
    return x + vit.mlp(vit.ln_2(x))

  def forward(self, frame, video_state, video_reset, point_state, point_ctl):
    """frame [1, 3, R, R] RGB 0..255; see the module docstring for the rest."""
    m = self.m
    v = self.num_video
    img = frame * (1.0 / 127.5) - 1.0
    vt = m.lin_proj(img).flatten(2)[0].transpose(0, 1) + self.video_pos  # [V, C]
    pt = self._point_tokens(point_ctl)  # [N, C]
    vr = video_reset.reshape(1, 1)
    vkeep = 1.0 - vr
    pkeep = 1.0 - torch.maximum(point_ctl[:, 3:4], vr)
    vs_out, ps_out = [], []
    last = len(m.blocks) - 1
    for i, blk in enumerate(m.blocks):
      vt, vs = self._ssm(blk.ssm_block, vt, video_state[i], vkeep)
      pt, ps = self._ssm(blk.ssm_block, pt, point_state[:, i].transpose(0, 1), pkeep)
      vs_out.append(vs)
      ps_out.append(ps.transpose(0, 1))
      if i < last:
        x = blk.vit_block(torch.cat([vt, pt], 0).unsqueeze(0))[0]
        vt, pt = x[:v], x[v:]
      else:  # last block: the video tokens' outputs are not used, only the points need queries
        pt = self._vit_points_only(blk.vit_block, vt, pt)
    pts = m.encoder_norm(pt)
    return self._heads(pts), torch.stack(vs_out, 0), torch.stack(ps_out, 1)


def load_tapnextpp(tapnet_root, weights, image_size=MODEL_SPACE):
  """Official TAPNext with the TAPNext++ weights (Lightning .ckpt or a plain state dict)."""
  if tapnet_root:
    sys.path.insert(0, os.path.abspath(tapnet_root))
  from tapnet.tapnext.tapnext_torch import TAPNext  # pylint: disable=g-import-not-at-top
  model = TAPNext(image_size=(image_size, image_size))
  ckpt = torch.load(weights, map_location='cpu', weights_only=True)
  sd = ckpt.get('state_dict', ckpt)
  sd = {k.removeprefix('tapnext.'): v for k, v in sd.items()}
  model.load_state_dict(sd)
  model.eval()
  for p in model.parameters():
    p.requires_grad = False
  return model


def zero_state(step, n_points, device='cpu'):
  v = step.num_video
  return (torch.zeros(NUM_LAYERS, PLANES, v, WIDTH, device=device),
          torch.zeros(n_points, NUM_LAYERS, PLANES, WIDTH, device=device))
