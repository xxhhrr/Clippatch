# src/pipelines/infer_ae_direct.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.utils.cfg import load_config
from src.compress.semantic_packer import create_semantic_packer
from src.recon.sr_runner import create_sr_runner


def _smart_load(model: torch.nn.Module, ckpt_path: str):
    """加载 ckpt 到 model，兼容 {state_dict:...} 或 DataParallel 的 'module.' 前缀。"""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")


def _param_sum(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().abs().sum().cpu())
    return s


def set_bn_no_running(model: torch.nn.Module):
    """让 BN 与训练时一致：不使用/维护 running_mean/var，一直用当前 batch 统计。"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def main():
    ap = argparse.ArgumentParser("AE-direct test (encoder->decoder, BN aligned to training)")
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--image", required=True, help="input .jpg/.png")
    ap.add_argument("--weights", default=None,
                    help="(optional) ckpt path to load into BOTH encoder/decoder")
    ap.add_argument("--show", action="store_true", help="matplotlib show")
    ap.add_argument("--save", default=None, help="save reconstructed image path")
    ap.add_argument("--no-fake-batch", action="store_true",
                    help="disable fake batch (by default we duplicate a flipped image to stabilize BN)")
    args = ap.parse_args()

    print("[MODE] AE-direct (no quant, no VGA, no zlib) | BN=no_running_stats, model.train()")

    cfg = load_config(args.config)

    # 构造两端模型（内部会按 YAML 尝试加载各自 weights；下方 --weights 可覆盖）
    packer = create_semantic_packer(cfg["compress"])
    srdec  = create_sr_runner(cfg["recon"])

    # （可选）命令行权重强制覆盖，确保两端用同一份
    if args.weights:
        _smart_load(packer.model, args.weights)
        _smart_load(srdec.model,  args.weights)

    # 关键：让推理端 BN 行为与训练一致（不使用 running stats），并切换到 train() 模式
    set_bn_no_running(packer.model)
    set_bn_no_running(srdec.model)
    packer.model.train()
    srdec.model.train()

    # 打印两端参数和校验是否一致（避免“看似加载了，其实不同”）
    ps_p = _param_sum(packer.model)
    ps_s = _param_sum(srdec.model)
    print(f"[debug] param_sum packer={ps_p:.6f}  sr={ps_s:.6f}  (diff={abs(ps_p-ps_s):.6f})")

    # 读取原图（保持原分辨率），注意：训练是 0..1，这里也使用 0..1
    img_path = Path(args.image); assert img_path.exists(), f"not found: {img_path}"
    orig_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert orig_bgr is not None, f"cv2 read failed: {img_path}"
    H, W = orig_bgr.shape[:2]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    x = orig_rgb.astype(np.float32) / 255.0                  # 0..1
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]         # [1,3,H,W]
    device = srdec.device if hasattr(srdec, "device") else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    x = x.to(torch.float32).to(device)

    # 可选：使用“假 batch”稳定 BN（默认开启；--no-fake-batch 可关）
    if args.no_fake_batch:
        x_in = x
    else:
        x_flip = torch.flip(x, dims=[3])      # 水平翻转
        x_in = torch.cat([x, x_flip], dim=0)  # [2,3,H,W]

    with torch.no_grad():
        y     = packer.model.encoder(x_in)
        x_hat = srdec.model.decoder(y, (H, W)).clamp(0.0, 1.0)
        x_hat = x_hat[:1]  # 只取第一张

        # 监控浮点范围（乘 255 前）
        print("x_hat float range:",
              float(x_hat.min().item()),
              float(x_hat.max().item()),
              float(x_hat.mean().item()))

    recon = (x_hat[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

    # PSNR（与原图 uint8 对比）
    val_psnr = psnr(orig_rgb, recon, data_range=255)
    print(f"[AE-direct] image={img_path.name}  PSNR={val_psnr:.2f} dB  size={W}x{H}")

    # 可视化
    if args.show:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(orig_rgb); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(recon);    axs[1].set_title(f"AE-Direct\nPSNR={val_psnr:.2f} dB")
        axs[1].axis("off")
        plt.tight_layout(); plt.show()

    # 保存
    if args.save:
        out_p = Path(args.save); out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
        print(f"[AE-direct] saved: {out_p}")


if __name__ == "__main__":
    main()
