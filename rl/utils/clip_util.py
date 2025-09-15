# -*- coding: utf-8 -*-
"""
Grad‑ECLIP heat‑map generator (memory‑safe, training‑friendly)
-------------------------------------------------------------
* Requirements: torch 2.x, torchvision, ftfy, regex, tqdm, git+https://github.com/openai/CLIP.git
* The function `get_heatmap(path, prompt, n_last_layers=1)` returns a 224×224 float32 NumPy array in [0,1].

Key change v2.1 ­– **use `@torch.no_grad()` instead of `@torch.inference_mode()`** for the
CLIP stem so that the returned tensor can later receive `requires_grad_(True)`.
This removes the runtime error:
```
RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.
```
while still keeping the early layers out of the computation graph.
"""

from __future__ import annotations
import gc, io
from contextlib import redirect_stdout
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                    InterpolationMode)

import clip  # pip install git+https://github.com/openai/CLIP.git

__all__ = ["get_heatmap", "print_gpu_tensors"]

device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, _ = clip.load("ViT-B/16", device=device)
clipmodel.eval()

_clip_inres = clipmodel.visual.input_resolution
_clip_ksize = clipmodel.visual.conv1.kernel_size

_transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def imgprocess(img: Image.Image, patch_size: Tuple[int, int] = (16, 16),
               scale_factor: float = 1.0) -> torch.Tensor:
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph
    img = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)(img).convert("RGB")
    return _transform(img)

# ---------------------------------------------------------------------------
# CLIP visual encoder – stem (no_grad) + tail (with_grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _clip_stem(x: torch.Tensor, n_last_layers: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Run conv + all ViT blocks except the last *n* layers without building a graph."""
    x = clipmodel.visual.conv1(x.half())
    h, w = x.shape[-2:]

    # flatten spatial → sequence & prepend class token
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = clipmodel.visual.class_embedding.to(x.dtype)
    x = torch.cat([cls + torch.zeros(x.shape[0], 1, cls.size(-1), dtype=x.dtype, device=x.device), x], dim=1)

    # resize positional embeddings
    pos = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos[:1], pos[1:]
    ph = _clip_inres // _clip_ksize[0]
    pw = _clip_inres // _clip_ksize[1]
    img_pos = img_pos.reshape(1, ph, pw, -1).permute(0, 3, 1, 2)
    img_pos = F.interpolate(img_pos, size=(h, w), mode="bicubic", align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    x = x + torch.cat((tok_pos[None], img_pos), dim=1)

    x = clipmodel.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD → LND

    if n_last_layers:
        blocks = clipmodel.visual.transformer.resblocks[:-n_last_layers]
        x = torch.nn.Sequential(*blocks)(x)

    return x.detach(), (h, w)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int = 1):
    L, B, D = q.shape
    d = D // heads
    q = q * (d ** -0.5)
    q = q.view(L, B * heads, d).transpose(0, 1)
    k = k.view(-1, B * heads, d).transpose(0, 1)
    v = v.view(-1, B * heads, d).transpose(0, 1)
    w = torch.bmm(q, k.transpose(1, 2)).softmax(dim=-1)
    out = torch.bmm(w, v).transpose(0, 1).contiguous().view(L, B, D)
    return out, w.view(B, heads, L, -1).mean(1)


def _clip_encode_dense(x: torch.Tensor, n_last_layers: int):
    stem, map_sz = _clip_stem(x, n_last_layers)
    stem.requires_grad_(True)

    tail = clipmodel.visual.transformer.resblocks[-n_last_layers:] if n_last_layers else []
    ql: List[torch.Tensor] = []
    kl: List[torch.Tensor] = []
    vl: List[torch.Tensor] = []
    al: List[torch.Tensor] = []

    linear = torch._C._nn.linear
    h = stem
    for blk in tail:
        h_in = h
        h = blk.ln_1(h_in)
        q, k, v = linear(h, blk.attn.in_proj_weight, blk.attn.in_proj_bias).chunk(3, dim=-1)
        attn, _ = _attention(q, k, v)
        h = linear(attn, blk.attn.out_proj.weight, blk.attn.out_proj.bias) + h_in
        h = h + blk.mlp(blk.ln_2(h))
        ql.append(q); kl.append(k); vl.append(v); al.append(attn)

    h = clipmodel.visual.ln_post(h.permute(1, 0, 2)) @ clipmodel.visual.proj
    return h, vl, ql, kl, al, map_sz

# ---------------------------------------------------------------------------
# Grad‑ECLIP core
# ---------------------------------------------------------------------------

def _sim(q: torch.Tensor, k: torch.Tensor):
    # 使用与main.py相同的归一化逻辑来减少噪声
    q_cls = F.normalize(q[:1, 0, :], dim=-1) 
    k_patch = F.normalize(k[1:, 0, :], dim=-1)
    
    cosine_qk = (q_cls * k_patch).sum(-1) 
    cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
    cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
    
    # Min-Max归一化，避免除零错误
    cosine_qk = (cosine_qk - cosine_qk_min) / (cosine_qk_max - cosine_qk_min + 1e-6)
    return cosine_qk


def _grad_eclip(cos: torch.Tensor, qs, ks, vs, attns, map_sz):
    heat = None
    for q, k, v, attn in zip(qs, ks, vs, attns):
        g, = torch.autograd.grad(cos, attn, retain_graph=True)
        layer_map = (g[:1, 0] * v[1:, 0] * _sim(q, k)[:, None]).sum(-1)
        heat = layer_map if heat is None else heat + layer_map
        del g, q, k, v, attn, layer_map
        torch.cuda.empty_cache()
    return F.relu_(heat).reshape(*map_sz)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_heatmap(img_path: str, prompt: str, n_last_layers: int = 1) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img_t = imgprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        txt_t = clip.tokenize([prompt]).to(device)
        txt_f = F.normalize(clipmodel.encode_text(txt_t), dim=-1)

    img_out, vs, qs, ks, atts, msz = _clip_encode_dense(img_t, n_last_layers)
    img_f = F.normalize(img_out[:, 0], dim=-1)
    cos = (img_f @ txt_f.T).squeeze()

    heat = _grad_eclip(cos, qs, ks, vs, atts, msz)
    heat = (heat - heat.min()) / (heat.max() + 1e-6)
    
    # 添加阈值处理，消除低激活噪声
    threshold = 0.1  # 可以根据需要调整
    heat = torch.where(heat < threshold, torch.zeros_like(heat), heat)
    # 重新归一化
    if heat.max() > 0:
        heat = (heat - heat.min()) / (heat.max() + 1e-6)

    with torch.no_grad():
        heat = F.interpolate(heat[None, None], size=(224, 224), mode="bicubic", align_corners=False)[0, 0]
    res = heat.cpu().numpy().astype(np.float32)

    del img_t, img_out, vs, qs, ks, atts, img_f, cos, heat, txt_f, txt_t
    torch.cuda.empty_cache()
    return res

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def print_gpu_tensors(context: str = "", log_file: str = "gpu_log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- GPU tensor dump ({context}) ---\n")
        mem = 0.0
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    m = o.element_size() * o.nelement() / 1048576.0
                    mem += m
                    f.write(f"  {type(o)} {tuple(o.size())} {m:.2f} MB\n")
            except Exception:
                pass
        f.write(f"Total: {mem:.2f} MB\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            torch.cuda.memory_summary(abbreviated=True)
        f.write(buf.getvalue() + "---\n\n")


