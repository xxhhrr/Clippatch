# src/recon/sr_runner.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import math, zlib
import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 与 packer 中模型结构保持一致 =====

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_stages=4):
        super().__init__()
        ch = base_ch
        layers = [ConvBlock(in_ch, ch)]
        for _ in range(num_stages):
            layers += [ConvBlock(ch, ch, k=3, s=2, p=1), ConvBlock(ch, ch)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
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
        a = torch.sigmoid(self.net(m_lr))
        return a.clamp_(0, 1)

class VGACodec(nn.Module):
    def __init__(self, img_ch=3, base_ch=64, num_stages=4, vga_ch=32, vga_layers=3):
        super().__init__()
        self.encoder = Encoder(in_ch=img_ch, base_ch=base_ch, num_stages=num_stages)
        self.decoder = Decoder(out_ch=img_ch, base_ch=base_ch, num_stages=num_stages)
        self.vga     = VGA(in_ch=1, base_ch=vga_ch, num_layers=vga_layers)

    @torch.no_grad()
    def reconstruct(self, payload: Dict[str, Any], mask_u8: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        H, W = map(int, payload["original_shape"])
        Hs, Ws, C = payload["latent_shape"]
        q_bg  = float(payload["qstep_bg"])
        q_roi = float(payload["qstep_roi"])
        tau   = float(payload["tau"])

        # 1) 解潜向量
        q_bytes = zlib.decompress(payload["latents_zlib"])
        q = np.frombuffer(q_bytes, dtype=np.int16).reshape(Hs, Ws, C).astype(np.float32)
        q = torch.from_numpy(q.transpose(2,0,1)).unsqueeze(0).to(device)  # 1,C,Hs,Ws

        # 2) 取低分辨率掩码
        arr = np.frombuffer(payload["mask_lr_png"], np.uint8)
        m_lr = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if m_lr is None or m_lr.shape[:2] != (Hs, Ws):
            # 回退：由原始 mask 下采样
            m_full = torch.from_numpy(mask_u8[None,None,...].astype(np.float32)).to(device)
            m_lr_t = F.interpolate(m_full, size=(Hs, Ws), mode="area")
            m_lr = (m_lr_t.squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255
        m_lr_t = torch.from_numpy((m_lr/255.0)[None,None,...].astype(np.float32)).to(device)
        m_lr_bin = (m_lr_t >= 0.5).float()

        # 3) VGA → α，构造 Δ，与编码端一致
        alpha = self.vga(m_lr_bin)
        log_bg, log_roi = math.log(q_bg), math.log(q_roi)
        delta = torch.exp(log_bg + alpha * (log_roi - log_bg)) * tau   # 1,1,Hs,Ws

        # 4) 反量化并重建
        y = q * delta
        x_hat = self.decoder(y, (H, W))
        x_hat = x_hat.clamp(0, 1.0) * 255.0
        x_hat = x_hat.squeeze(0).permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
        return x_hat

# =========================
#   Runner (public API)
# =========================

class SRRunner:
    """
    解复用 + 重构：与打包端的 VGACodec 对称
    """
    def __init__(self, config: Dict[str, Any]):
        neu = config.get("neural", {})
        self.num_stages   = int(neu.get("num_stages", 4))
        self.base_ch      = int(neu.get("base_channels", 64))
        self.vga_ch       = int(neu.get("vga_channels", 32))
        self.vga_layers   = int(neu.get("vga_layers", 3))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGACodec(img_ch=3, base_ch=self.base_ch,
                              num_stages=self.num_stages,
                              vga_ch=self.vga_ch, vga_layers=self.vga_layers).to(self.device).eval()
        # 可选加载相同权重
        weights = neu.get("weights", None)
        if weights:
            try:
                ckpt = torch.load(weights, map_location=self.device)
                sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                self.model.load_state_dict(sd, strict=False)
            except Exception:
                pass

    def reconstruct_image(self, payload: Dict[str, Any]) -> np.ndarray:
        if payload.get("codec") != "vga_codec_v1":
            raise ValueError("Unsupported codec in payload; expected 'vga_codec_v1'")
        # 解码端需要同一张 mask（run_mvp_sr 已持有原 mask）
        # 但我们发送了 mask_lr_png 作副信息，足以复现 VGA 引导
        # 这里传入一个“单位掩码”仅作回退（一般用不到）
        H, W = map(int, payload["original_shape"])
        mask_fallback = np.ones((H, W), np.uint8)
        return self.model.reconstruct(payload, mask_fallback)

def create_sr_runner(config: Dict[str, Any]) -> SRRunner:
    return SRRunner(config)
