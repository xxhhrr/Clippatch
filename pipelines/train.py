# -*- coding: utf-8 -*-
from __future__ import annotations
import math, logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.cfg import load_config
from src.recon.sr_runner import VGACodec
from src.locate.clip_locator import create_clip_locator
from src.segment.sam_runner import create_sam_runner
from src.data.data_collector import build_id_text_list  # ← 你刚加入的收集器
from utils.utils import get_inst_bbox
import torchvision.utils as vutils
import zlib


SAVE_DIR = Path("outputs/train_snap"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Round STE（量化直通估计） -----------------
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g

def ste_round(x):  # 便捷包装
    return _RoundSTE.apply(x)

# ----------------- 数据集（按 index 端到端在线生成 mask） -----------------
class E2EIndexDataset(Dataset):
    """
    传入 build_id_text_list 返回的 index；在 __getitem__ 里：
      读图 → CLIP 定位 → SAM 分割 → 得到 mask（失败则 bbox 回退）
    为避免 DataLoader 多进程下复制 GPU 句柄，这里用“懒加载”：
      每个 worker 第一次取样时再各自构造 locator/sam 实例。
    """
    def __init__(
        self,
        index: List[Dict[str, Any]],
        locator_cfg: Dict[str, Any],
        sam_cfg: Dict[str, Any],
        resize: Optional[tuple[int,int]] = None,  # (W,H)
        save_vis: bool = False,
        vis_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.index = index
        self.locator_cfg = locator_cfg
        self.sam_cfg = sam_cfg
        self.resize = resize
        self.save_vis = save_vis
        self.vis_dir = Path(vis_dir) if vis_dir else None
        if self.save_vis and self.vis_dir:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.to_tensor = transforms.ToTensor()
        # worker 本地对象
        self._locator = None
        self._sam = None

    def _ensure_models(self):
        if self._locator is None:
            self._locator = create_clip_locator(self.locator_cfg)
        if self._sam is None:
            self._sam = create_sam_runner(self.sam_cfg)

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        self._ensure_models()
        rec = self.index[i]
        img_path = Path(rec["image_path"])
        text = rec.get("text", "") or ""

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读图失败: {img_path}")
        H0, W0 = bgr.shape[:2]

        if self.resize:
            bgr = cv2.resize(bgr, self.resize, interpolation=cv2.INTER_AREA)
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 定位
        # out_loc = self._locator.locate(rgb, text, out_size=(W, H), array_mode="rgb")
        out_loc = get_inst_bbox(rec.get("inst_path"), rec.get("ann_id"), ori_size=(W0, H0), out_size=(W, H))

        if not out_loc:
            out_loc = {'bbox':[0,0,W,H], 'score':0.0}

        # SAM（优先 array 接口）
        if hasattr(self._sam, "segment_from_bbox_array"):
            out_sam = self._sam.segment_from_bbox_array(rgb, out_loc)
        else:
            out_sam = self._sam.segment_from_bbox(str(img_path), out_loc, save_visualization=False)

        bbox = out_sam.get("bbox") or out_loc.get("bbox") or [0,0,W,H]
        # 生成/回退 mask
        m = out_sam.get("mask", None)
        if m is None or np.asarray(m).sum()==0:
            if isinstance(bbox, dict) and {"x1","y1","x2","y2"} <= set(bbox.keys()):
                x1,y1,x2,y2 = bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]
            else:
                x,y,w,h = bbox
                x1,y1,x2,y2 = x, y, x+max(1,w), y+max(1,h)
            m = np.zeros((H,W), np.uint8); m[y1:y2, x1:x2] = 1
        else:
            m = m.astype(np.uint8)
            if m.shape[:2] != (H,W):
                m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)

        # 可视化（可选）
        if self.save_vis and self.vis_dir:
            vis = rgb.copy()
            vis[m>0] = (0.5*vis[m>0] + 0.5*np.array([0,255,0],dtype=np.uint8)).astype(np.uint8)
            cv2.imwrite(str(self.vis_dir / f"{img_path.stem}_mask_vis.jpg"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        x = self.to_tensor(rgb).to(torch.float32)  ## (3,H,W) 0..1
        m01 = torch.from_numpy(m.astype(np.float32))[None]    # (1,H,W)
        return {"image": x, "mask": m01, "name": img_path.name, "text": text}


@torch.no_grad()
def eval_true_bpp_psnr(
    net: VGACodec,
    x: torch.Tensor,        # [B,3,H,W] in [0,1]
    m: torch.Tensor,        # [B,1,H,W] in {0,1}
    q_bg: float, q_roi: float, tau: float,
    zlib_level: int = 6,
):
    """
    用真实 round + zlib 统计比特，并用反量化后 decoder 的结果计算“真 PSNR”。
    返回: (bpp, psnr_true)
    计入: latents_zlib + mask_lr_png 的比特数
    """
    device = x.device
    B, _, H, W = x.shape

    # 1) encoder
    y = net.encoder(x)  # [B,C,Hs,Ws]
    _, C, Hs, Ws = y.shape

    # 2) 掩码下采样到潜空间 + VGA -> alpha -> Δ（与推理端一致）
    m_lr = F.interpolate(m, size=(Hs, Ws), mode="area")
    m_bin = (m_lr >= 0.5).float()
    alpha = net.vga(m_bin)
    log_bg, log_roi = math.log(q_bg), math.log(q_roi)
    delta = torch.exp(log_bg + alpha * (log_roi - log_bg)) * float(tau)  # [B,1,Hs,Ws]

    # 3) 真实量化（round）→ zlib 压缩；同时生成 mask_lr_png 字节
    total_bits = 0
    for b in range(B):
        q_b = torch.round(y[b:b+1] / delta[b:b+1])  # [1,C,Hs,Ws]
        q_np = q_b.squeeze(0).permute(1, 2, 0).contiguous().to(torch.int16).cpu().numpy()  # [Hs,Ws,C]
        lat_bytes = zlib.compress(q_np.tobytes(order="C"), level=zlib_level)

        # mask 副信息（与 packer.compress 对齐）
        mask_lr_u8 = (m_bin[b].squeeze(0).cpu().numpy() * 255).astype(np.uint8)  # [Hs,Ws], 0/255
        ok, buf = cv2.imencode('.png', mask_lr_u8)
        if not ok:
            raise RuntimeError("mask_lr png encode failed in eval")

        total_bits += (len(lat_bytes) + len(buf)) * 8

    bpp = total_bits / float(B * H * W)

    # 4) 反量化并重建（模拟 decompress 后的 decoder）
    q = torch.round(y / delta)
    y_hat = q * delta
    x_hat_true = net.decoder(y_hat, (H, W)).clamp(0, 1)

    # 5) 真 PSNR（0-1 尺度）
    mse_val = F.mse_loss(x_hat_true, x).item()
    psnr_true = 10.0 * math.log10(1.0 / max(mse_val, 1e-12))

    return bpp, psnr_true


# ----------------- 训练前向（兼容无 forward_train 的 VGACodec） -----------------
def codec_forward_train(
    net: VGACodec,
    x: torch.Tensor,          # (B,3,H,W) 0/1
    m: torch.Tensor,          # (B,1,H,W) 0/1
    use_quant: bool,
    q_roi: float, q_bg: float, tau: float
):
    """
    不依赖 VGACodec 是否实现 forward_train：
      encoder → (VGA→Δ→可选量化) → decoder
    """
    y = net.encoder(x)                        # 1) 编码
    
    Hs, Ws = y.shape[-2:]
    m_lr = F.interpolate(m, size=(Hs, Ws), mode="area")
    m_bin = (m_lr >= 0.5).float()
    alpha = net.vga(m_bin)                    # 2) VGA→alpha
    log_bg, log_roi = math.log(q_bg), math.log(q_roi)
    delta = torch.exp(log_bg + alpha * (log_roi - log_bg)) * float(tau)

    if use_quant:
        q = ste_round(y / delta)
        yq = q * delta
    else:
        yq = y

    H, W = x.shape[-2:]
    x_hat = net.decoder(yq, (H, W))           # 3) 解码
    x_hat = x_hat.clamp(0.0, 1.0)
    return x_hat, {"alpha": alpha, "delta": delta}

# ----------------- 主训练 -----------------
def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    log = logging.getLogger("train_vga_codec_e2e_index")

    cfg = load_config("configs/mvp.yaml")

    # 路径 & 样本索引（满足：有 text/<id>.txt 且 instances/<id>.txt）
    paths = cfg["paths"]
    images_dir = Path(paths["images"])
    text_dir   = Path(paths["text"])
    inst_dir   = Path(cfg.get("paths", {}).get("instances",
                    cfg.get("dataset", {}).get("instances_dir", "data/instances")))
    desired    = int(cfg.get("dataset", {}).get("desired_count", 20000))
    save_idx   = Path(cfg.get("dataset", {}).get("save_index", "outputs/datalist/train_index.jsonl"))

    index = build_id_text_list(images_dir, text_dir, inst_dir, limit=desired, save_list_path=save_idx)
    if not index:
        raise RuntimeError("没有满足条件的样本（需同时存在 text/<id>.txt 与 instances/<id>.txt）")
    log.info(f"训练样本数: {len(index)}  | images={images_dir}  text={text_dir}  inst={inst_dir}")

    # 训练超参
    tcfg = cfg.get("train", {})
    EPOCHS      = int(tcfg.get("epochs", 2))
    BATCH_SIZE  = int(tcfg.get("batch_size", 4))
    LR          = float(tcfg.get("lr", 1e-4))
    WD          = float(tcfg.get("weight_decay", 1e-4))
    NUM_WORKERS = int(tcfg.get("num_workers", 0))  # 若>0，每个 worker 会懒加载一份 CLIP/SAM
    IMG_SIZE    = tcfg.get("train_size", [512,512])  # [W,H]
    USE_QUANT   = bool(tcfg.get("use_quant", False))
    ROI_WEIGHT  = float(tcfg.get("roi_weight", 1.0))
    SAVE_DIR    = Path(tcfg.get("save_dir","checkpoints")); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_NAME   = str(tcfg.get("save_name","vga_e2e.pth"))

    # 编码器/解码器/VGA 参数
    ncfg = cfg.get("neural", {})
    QSTEP_ROI = float(ncfg.get("qstep_roi", 2.0))
    QSTEP_BG  = float(ncfg.get("qstep_bg", 6.0))
    TAU_TR    = float(ncfg.get("tau_train", 1.0))

    # DataLoader
    resize = (int(IMG_SIZE[0]), int(IMG_SIZE[1])) if IMG_SIZE else None
    dl = DataLoader(
        E2EIndexDataset(
            index=index,
            locator_cfg=cfg["locate"],
            sam_cfg=cfg["segment"],
            resize=resize,
            save_vis=bool(tcfg.get("save_vis", False)),
            vis_dir=Path(cfg["paths"].get("vis_dir","outputs/vis"))
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VGACodec(
        img_ch=3,
        base_ch=int(ncfg.get("base_channels", 64)),
        num_stages=int(ncfg.get("num_stages", 4)),
        vga_ch=int(ncfg.get("vga_channels", 32)),
        vga_layers=int(ncfg.get("vga_layers", 3)),
    ).to(device).train()

    # （可选）载入已有权重继续训
    weights = ncfg.get("weights", "")
    if weights and Path(weights).exists():
        sd = torch.load(weights, map_location=device)
        sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        net.load_state_dict(sd, strict=False)
        log.info(f"Loaded weights: {weights}")


    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)

    global_step = 0
    for ep in range(1, EPOCHS+1):
        for batch in dl:
            x = batch["image"].to(device)   # (B,3,H,W) 0/1
            m = batch["mask"].to(device)    # (B,1,H,W) 0/1

            # 训练前向（兼容无 forward_train 的 VGACodec）
            x_hat, _ = codec_forward_train(
                net, x, m, use_quant=USE_QUANT,
                q_roi=QSTEP_ROI, q_bg=QSTEP_BG, tau=TAU_TR
            )

            # MSE（可选 ROI 加权）
            if ROI_WEIGHT > 1.0:
                w = torch.ones_like(m) + (ROI_WEIGHT - 1.0) * m
                mse = ((x_hat - x)**2 * w).mean()
            else:
                mse = F.mse_loss(x_hat, x)
            loss = mse

            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % 50 == 0:
                with torch.no_grad():
                    PIX_MAX = 1.0
                    mse_val = F.mse_loss(x_hat, x).item()
                    psnr = 10.0 * math.log10((PIX_MAX**2) / max(mse_val, 1e-12))
                    xhat_vis = x_hat.detach().clamp(0,1).cpu()  # (B,3,H,W) [0,1]
                    # 批量保存成一张网格图：
                    vutils.save_image(xhat_vis, SAVE_DIR / f"xhat_{global_step:07d}.png")
                log.info(f"[ep {ep}] step {global_step} | loss={loss.item():.4f} | psnr≈{psnr:.2f}dB | quant={USE_QUANT}")


        # —— 在每个 epoch 结束时，抽一小批样本做真实 bpp/PSNR 评估 ——
        # 这里用“刚才最后一个 batch”的 x、m；也可以单独准备一个小的 val DataLoader。
        with torch.no_grad():
            # 为了评估更稳，可以只取前 N 张（比如 4 张），避免太慢
            N = min(4, x.shape[0])
            bpp_true, psnr_true = eval_true_bpp_psnr(
                net,
                x[:N], m[:N],
                q_bg=QSTEP_BG, q_roi=QSTEP_ROI, tau=TAU_TR,
                zlib_level=6
            )
        log.info(f"[epoch {ep}] true_bpp={bpp_true:.4f} | true_psnr={psnr_true:.2f} dB")

        # 每个 epoch 存一次
        ep_path = SAVE_DIR / f"{SAVE_NAME.replace('.pth','')}_ep{ep}.pth"
        torch.save(net.state_dict(), ep_path)
        log.info(f"saved: {ep_path}")

    final_path = SAVE_DIR / SAVE_NAME
    torch.save(net.state_dict(), final_path)
    log.info(f"final saved: {final_path}")

if __name__ == "__main__":
    main()
