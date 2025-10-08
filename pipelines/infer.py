# src/pipelines/infer_recon.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.utils.cfg import load_config
from src.segment.sam_runner import create_sam_runner
from src.locate.clip_locator import create_clip_locator
from src.compress.semantic_packer import create_semantic_packer
from src.recon.sr_runner import create_sr_runner
from utils.utils import get_inst_bbox
from src.data.data_collector import read_first_sent_ann_id

# ---------------- helpers ----------------
_ID_PAT = re.compile(r"(\d{12})")
def image_id_from_name(name: str) -> str | None:
    m = _ID_PAT.search(name)
    return m.group(1) if m else None


def xywh_to_xyxy(b):
    x,y,w,h = [float(v) for v in b]
    return [int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h))]

def rect_mask(h, w, xyxy):
    x1,y1,x2,y2 = xyxy
    m=np.zeros((h,w), np.uint8)
    x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
    x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
    m[y1:y2, x1:x2]=1
    return m

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Single-image inference (pack->recon)")
    ap.add_argument("--config", default="configs/mvp.yaml")
    ap.add_argument("--image", required=True, help="path to an input image (.jpg)")
    ap.add_argument("--text", default=None, help="optional prompt; fallback to text/<id>.txt")
    ap.add_argument("--ann-id", type=int, default=None, help="prefer instance with this ann_id")
    ap.add_argument("--no-sam", action="store_true", help="use rectangle mask from bbox (skip SAM)")
    ap.add_argument("--target-bpp", type=float, default=None, help="override pipeline.target_bpp")
    ap.add_argument("--bpp-tol", type=float, default=None, help="override pipeline.bpp_tolerance")
    ap.add_argument("--show", action="store_true", help="show window via matplotlib")
    ap.add_argument("--save", default=None, help="optional path to save reconstructed image")
    args = ap.parse_args()

    cfg = load_config(args.config)
    img_path = Path(args.image)
    assert img_path.exists(), f"image not found: {img_path}"

    # read original (keep original size)
    orig_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert orig_bgr is not None, f"cv2 read failed: {img_path}"
    H, W = orig_bgr.shape[:2]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_id = image_id_from_name(img_path.name)
    # text
    text = args.text
    if not text:
        text_dir = Path(cfg["paths"]["text"])
        text, ann_id = read_first_sent_ann_id(text_dir / f"{img_id}.txt") or ""

    # bbox: prefer instances/<id>.txt
    inst_dir = Path(cfg["paths"]["instances"])
    bbox_xyxy = None
    if inst_dir.exists():
        b = get_inst_bbox(inst_dir / f"{img_id}.txt", args.ann_id, (W, H), (512, 512))
        if b is not None:
            bbox_xyxy = xywh_to_xyxy(b)

    locator = None
    if bbox_xyxy is None:
        locator = create_clip_locator(cfg["locate"])
        out_loc = locator.locate(str(img_path), text, out_size=(W, H))
        bb = out_loc.get("bbox", [0,0,W,H])
        # 兼容 xywh
        if len(bb)==4:
            x,y,w,h = bb
            bbox_xyxy = [int(x),int(y),int(x+w),int(y+h)]
        else:
            bbox_xyxy = bb

    # mask: SAM or rectangle
    if args.no_sam:
        mask = rect_mask(H, W, bbox_xyxy)
    else:
        sam = create_sam_runner(cfg["segment"])
        out_sam = sam.segment_from_bbox(str(img_path), {"bbox": bbox_xyxy}, save_visualization=False)
        mask = out_sam.get("mask", None)
        if mask is None or np.asarray(mask).sum()==0:
            mask = rect_mask(H, W, bbox_xyxy)

    # pack & reconstruct
    packer = create_semantic_packer(cfg["compress"])
    srdec  = create_sr_runner(cfg["recon"])

    tgt_bpp = float(args.target_bpp if args.target_bpp is not None else cfg["pipeline"].get("target_bpp", 1.0))
    bpp_tol = float(args.bpp_tol  if args.bpp_tol  is not None else cfg["pipeline"].get("bpp_tolerance", 0.02))
    target_bits = int(tgt_bpp * W * H)
    bits_tol    = int(bpp_tol * target_bits)

    payload = packer.pack_image(str(img_path), mask.astype(np.uint8), bbox_xyxy,
                                target_bits=target_bits, bits_tolerance=bits_tol)
    recon = srdec.reconstruct_image(payload)  # RGB uint8

    # PSNR
    val_psnr = psnr(orig_rgb, recon, data_range=255)
    print(f"[infer] image={img_path.name}  PSNR={val_psnr:.2f} dB  bpp={payload['bpp']:.3f}")

    # show (matplotlib, safer than cv2.imshow on headless)
    if args.show:
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2, figsize=(10,5))
        axs[0].imshow(orig_rgb); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(recon);    axs[1].set_title(f"Reconstructed\nPSNR={val_psnr:.2f} dB, bpp={payload['bpp']:.3f}")
        axs[1].axis("off")
        plt.tight_layout(); plt.show()

    # save
    if args.save:
        save_p = Path(args.save)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_p), cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
        print(f"[infer] saved: {save_p}")

if __name__ == "__main__":
    main()
