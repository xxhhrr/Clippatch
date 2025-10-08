# src/compress/semantic_packer.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json, math, zlib
import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
#   Model building blocks
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Encoder(nn.Module):
    """
    可训练分析变换：多层 stride=2 卷积降采样
    输出潜向量 y ∈ R^{B×C×Hs×Ws}
    """
    def __init__(self, in_ch=3, base_ch=64, num_stages=4):
        super().__init__()
        ch = base_ch
        layers = [ConvBlock(in_ch, ch)]
        for _ in range(num_stages):
            layers += [ConvBlock(ch, ch, k=3, s=2, p=1), ConvBlock(ch, ch)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """
    可训练合成变换：多层反卷积上采样回原分辨率
    """
    def __init__(self, out_ch=3, base_ch=64, num_stages=4):
        super().__init__()
        ch = base_ch
        ups = []
        for _ in range(num_stages):
            ups += [
                nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                ConvBlock(ch, ch)
            ]
        self.up = nn.Sequential(*ups)
        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1)
    def forward(self, y, out_hw: Tuple[int,int]):
        x = self.up(y)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.head(x)
        return x

class VGA(nn.Module):
    """
    VGA：仅以掩码为输入，输出与潜空间同分辨率的 α ∈ [0,1]
    （训练时学到“压缩率图”，非 SFT γ/β）
    """
    def __init__(self, in_ch=1, base_ch=32, num_layers=3):
        super().__init__()
        layers = []
        ch = base_ch
        layers.append(ConvBlock(in_ch, ch))
        for _ in range(num_layers-2):
            layers.append(ConvBlock(ch, ch))
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)
    def forward(self, m_lr):
        # 输出到 [0,1]，初始更偏向于“mask=1 的区域更高 α”
        a = torch.sigmoid(self.net(m_lr))
        return a.clamp_(0, 1)

class VGACodec(nn.Module):
    """
    端到端可训练编解码器（不含熵模型），用 α 控制空间可变量化步长 Δ。
    Δ(x,y) = tau * exp( log(Δ_bg) + α(x,y) * (log(Δ_roi) - log(Δ_bg)) )
    """
    def __init__(self, img_ch=3, base_ch=64, num_stages=4, vga_ch=32, vga_layers=3):
        super().__init__()
        self.encoder = Encoder(in_ch=img_ch, base_ch=base_ch, num_stages=num_stages)
        self.decoder = Decoder(out_ch=img_ch, base_ch=base_ch, num_stages=num_stages)
        self.vga     = VGA(in_ch=1, base_ch=vga_ch, num_layers=vga_layers)

    @torch.no_grad()
    def compress(self, x_u8: np.ndarray, mask_u8: np.ndarray,
                 q_roi: float, q_bg: float, tau: float, zlib_level: int = 6) -> Dict[str, Any]:
        """
        x_u8: H×W×3 RGB, uint8
        mask_u8: H×W, {0,1}
        返回：{"latents_zlib": bytes, "mask_lr_png": bytes, "latent_shape": (Hs,Ws,C)}
        """
        device = next(self.parameters()).device
        H, W = x_u8.shape[:2]

        # 1) 张量化
        x = torch.from_numpy(x_u8.transpose(2,0,1)).unsqueeze(0).float().to(device)  # 1,3,H,W
        x = x / 255.0
        m = torch.from_numpy(mask_u8[None,None,...].astype(np.float32)).to(device)   # 1,1,H,W

        # 2) 编码得到潜向量 y；下采样掩码到潜空间 m_lr
        y = self.encoder(x)                                  # 1,C,Hs,Ws
        _, _, Hs, Ws = y.shape
        m_lr = F.interpolate(m, size=(Hs, Ws), mode="area")  # 1,1,Hs,Ws
        m_lr_bin = (m_lr >= 0.5).float()

        # 3) VGA → α；构造 Δ 图（对每通道共享）
        alpha = self.vga(m_lr_bin)                           # 1,1,Hs,Ws
        log_bg, log_roi = math.log(q_bg), math.log(q_roi)
        delta = torch.exp(log_bg + alpha * (log_roi - log_bg)) * float(tau)  # 1,1,Hs,Ws

        # 4) 量化：q = round(y / Δ)，通道共享 Δ
        q = torch.round(y / delta)

        # 5) 序列化：zlib 压缩 q（int16）
        q_np = q.squeeze(0).permute(1,2,0).contiguous().cpu().numpy().astype(np.int16)  # Hs,Ws,C
        lat_bytes = zlib.compress(q_np.tobytes(order="C"), level=zlib_level)

        # 6) 掩码副信息（Hs×Ws），PNG 存储（开销极小）
        mask_lr_u8 = (m_lr_bin.squeeze().cpu().numpy() * 255).astype(np.uint8)
        ok, buf = cv2.imencode('.png', mask_lr_u8)
        if not ok: raise RuntimeError("mask_lr png encode failed")
        mask_bytes = buf.tobytes()

        return {
            "latents_zlib": lat_bytes,
            "mask_lr_png": mask_bytes,
            "latent_shape": (Hs, Ws, y.shape[1])
        }

    @torch.no_grad()
    def decompress(self, payload: Dict[str, Any], mask_u8: np.ndarray,
                   q_roi: float, q_bg: float, tau: float) -> np.ndarray:
        """
        根据位流与掩码复现 Δ，反量化并重建 RGB uint8
        """
        device = next(self.parameters()).device
        H, W = map(int, payload["original_shape"])
        Hs, Ws, C = payload["latent_shape"]

        # 1) 解压潜向量 q
        q_bytes = zlib.decompress(payload["latents_zlib"])
        q = np.frombuffer(q_bytes, dtype=np.int16).reshape(Hs, Ws, C).astype(np.float32)
        q = torch.from_numpy(q.transpose(2,0,1)).unsqueeze(0).to(device)  # 1,C,Hs,Ws

        # 2) 解码 mask_lr
        arr = np.frombuffer(payload["mask_lr_png"], np.uint8)
        m_lr = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)      # Hs,Ws uint8
        if m_lr is None:
            # 如未随流发送，回退由原始 mask 下采样
            m_full = torch.from_numpy(mask_u8[None,None,...].astype(np.float32)).to(device)
            m_lr_t = F.interpolate(m_full, size=(Hs, Ws), mode="area")
            m_lr = (m_lr_t.squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255
        m_lr_t = torch.from_numpy((m_lr/255.0)[None,None,...].astype(np.float32)).to(device)
        m_lr_bin = (m_lr_t >= 0.5).float()

        # 3) VGA → α；构造 Δ，与编码端一致
        alpha = self.vga(m_lr_bin)                           # 1,1,Hs,Ws
        log_bg, log_roi = math.log(q_bg), math.log(q_roi)
        delta = torch.exp(log_bg + alpha * (log_roi - log_bg)) * float(tau)  # 1,1,Hs,Ws

        # 4) 反量化并重建
        y = q * delta                                        # 1,C,Hs,Ws
        x_hat = self.decoder(y, (H, W))                      # 1,3,H,W
        x_hat = x_hat.clamp(0, 255).squeeze(0).permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
        return x_hat

# =========================
#   Utils
# =========================

def _bbox_to_xyxy(bbox, W, H):
    if bbox is None:
        x1,y1,x2,y2 = 0,0,W,H
    elif isinstance(bbox, dict) and {"x1","y1","x2","y2"} <= set(bbox.keys()):
        x1,y1,x2,y2 = int(bbox["x1"]),int(bbox["y1"]),int(bbox["x2"]),int(bbox["y2"])
    elif isinstance(bbox, dict) and "bbox" in bbox:
        x,y,w,h = [int(round(v)) for v in bbox["bbox"]]
        x1,y1,x2,y2 = x, y, x+max(1,w), y+max(1,h)
    else:
        x,y,w,h = [int(round(v)) for v in bbox]
        if w>0 and h>0: x1,y1,x2,y2 = x, y, x+max(1,w), y+max(1,h)
        else:           x1,y1,x2,y2 = x, y, max(x+1,w), max(y+1,h)
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(1, min(x2, W));   y2 = max(1, min(y2, H))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    return x1,y1,x2,y2

def _encode_png_u8(gray_u8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', gray_u8)
    if not ok: raise RuntimeError("png encode failed")
    return buf.tobytes()

# =========================
#   Packer (public API)
# =========================

class SemanticPacker:
    """
    可训练 VGA 编解码 + 精确 bpp 控码（二分 tau）
    发送：zlib 压缩潜向量 + 低分辨率 mask（PNG）
    """
    def __init__(self, config: Dict[str, Any]):
        neu = config.get("neural", {})
        self.num_stages   = int(neu.get("num_stages", 4))
        self.base_ch      = int(neu.get("base_channels", 64))
        self.vga_ch       = int(neu.get("vga_channels", 32))
        self.vga_layers   = int(neu.get("vga_layers", 3))

        self.qstep_roi    = float(neu.get("qstep_roi", 2.0))
        self.qstep_bg     = float(neu.get("qstep_bg", 6.0))

        # 控码范围（bpp 0.1~1.5 对应 tau 搜索范围，可按需调）
        self.tau_min      = float(neu.get("tau_min", 0.25))
        self.tau_max      = float(neu.get("tau_max", 8.0))
        self.search_iters = int(neu.get("search_iters", 8))

        self.zlib_level   = int(neu.get("zlib_level", 6))

        outdir = config.get('output_dir') or config.get('paths.compress_output') or 'outputs/payloads'
        self.output_dir = Path(outdir); self.output_dir.mkdir(parents=True, exist_ok=True)

        # 构建模型并加载权重（可训练）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGACodec(img_ch=3, base_ch=self.base_ch,
                              num_stages=self.num_stages,
                              vga_ch=self.vga_ch, vga_layers=self.vga_layers).to(self.device).eval()
        # 可选权重
        weights = neu.get("weights", None)
        if weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location=self.device)
            # 支持保存的 {state_dict: ...} 或直接 state_dict
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd, strict=False)

        # 兼容旧字段（不使用，仅避免报错）
        self.margin_ratio = float(config.get('margin_ratio', 0.10))

    def pack_image(
        self,
        image_path: str,
        mask: np.ndarray,
        bbox,
        target_bits: Optional[int] = None,
        bits_tolerance: Optional[int] = None
    ) -> Dict[str, Any]:

        # 读图
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError(f"Cannot load image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 掩码二值化
        mask01 = (mask.astype(np.uint8) > 0).astype(np.uint8)

        # 速率控制：二分 tau，使总比特 ≈ target_bits
        if target_bits is None or bits_tolerance is None:
            tau = 1.0
            enc = self.model.compress(rgb, mask01, self.qstep_roi, self.qstep_bg, tau, self.zlib_level)
            lat_bytes = enc["latents_zlib"]
            mask_bytes = enc["mask_lr_png"]
        else:
            lo, hi = self.tau_min, self.tau_max
            best = None
            for _ in range(self.search_iters):
                mid = 0.5 * (lo + hi)
                enc = self.model.compress(rgb, mask01, self.qstep_roi, self.qstep_bg, mid, self.zlib_level)
                lat_b = enc["latents_zlib"]; mask_b = enc["mask_lr_png"]
                tot_bits = (len(lat_b) + len(mask_b)) * 8
                if best is None or abs(tot_bits - target_bits) < abs(best[2] - target_bits):
                    best = (lat_b, mask_b, tot_bits, mid, enc["latent_shape"])
                if tot_bits > target_bits + bits_tolerance:
                    lo = mid   # 比特过大，增大 tau（步长更大→更省比特）
                elif tot_bits < target_bits - bits_tolerance:
                    hi = mid   # 比特过小，减小 tau
                else:
                    best = (lat_b, mask_b, tot_bits, mid, enc["latent_shape"])
                    break
            lat_bytes, mask_bytes, _, tau, latent_shape = best

        # 统计与 payload
        lat_bits  = len(lat_bytes) * 8
        mask_bits = len(mask_bytes) * 8
        total_bits = lat_bits + mask_bits
        bpp = total_bits / (H * W)

        x1,y1,x2,y2 = _bbox_to_xyxy(bbox, W, H)

        payload = {
            "codec": "vga_codec_v1",
            "original_shape": [int(H), int(W)],
            "latent_shape": list(enc["latent_shape"] if target_bits is None else latent_shape),
            "num_stages": int(self.num_stages),
            "base_channels": int(self.base_ch),
            "vga_channels": int(self.vga_ch),
            "vga_layers": int(self.vga_layers),
            "qstep_bg": float(self.qstep_bg),
            "qstep_roi": float(self.qstep_roi),
            "tau": float(tau),
            "mask_lr_png": mask_bytes,
            "latents_zlib": lat_bytes,
            "roi_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bits": {"latents": lat_bits, "mask": mask_bits, "total": total_bits},
            "bpp": float(bpp)
        }

        self._save_payload(image_path, payload)
        return payload

    def _save_payload(self, image_path: str, payload: Dict[str, Any]) -> None:
        name = Path(image_path).stem
        out = self.output_dir
        (out / f"{name}_latents.bin.zlib").write_bytes(payload["latents_zlib"])
        (out / f"{name}_mask_lr.png").write_bytes(payload["mask_lr_png"])
        meta = {k: payload[k] for k in (
            "codec","original_shape","latent_shape","num_stages","base_channels",
            "vga_channels","vga_layers","qstep_bg","qstep_roi","tau",
            "roi_bbox_xyxy","bits","bpp"
        )}
        (out / f"{name}_payload_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def create_semantic_packer(config): 
    return SemanticPacker(config)
