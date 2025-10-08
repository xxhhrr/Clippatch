# src/pipelines/run_mvp_sr.py
from pathlib import Path
import logging, csv, random, numpy as np, torch, json
import cv2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from src.utils.cfg import load_config
from src.locate.clip_locator import create_clip_locator
from src.segment.sam_runner import create_sam_runner
from src.compress.semantic_packer import create_semantic_packer
from src.recon.sr_runner import create_sr_runner
from utils.utils import clean_outputs_keep_dirs
from src.data.data_collector import build_id_text_list  # ← 用这个收集样本

# ------------ helpers ------------
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# ------------ main ------------
def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("mvp")

    cfg = load_config("configs/mvp.yaml")
    set_seeds(cfg["pipeline"].get("seed", 42))

    # 码率控制
    TARGET_BPP = float(cfg["pipeline"].get("target_bpp", 1.0))
    BPP_TOL    = float(cfg["pipeline"].get("bpp_tolerance", 0.02))

    # 路径
    images_dir = Path(cfg["paths"]["images"])
    text_dir   = Path(cfg["paths"]["text"])
    inst_dir   = Path(cfg.get("paths", {}).get("instances", cfg.get("dataset", {}).get("instances_dir", "data/instances")))
    if not text_dir.exists():
        raise FileNotFoundError(f"text 目录不存在: {text_dir}")

    # 用数据收集器按顺序拿到前 N 个满足条件的样本（有 text 和 instances）
    desired_count = int(cfg.get("dataset", {}).get("desired_count", cfg["pipeline"].get("limit_images", 2000)))
    save_index    = Path(cfg.get("dataset", {}).get("save_index", "outputs/datalist/infer_index.jsonl"))
    index = build_id_text_list(
        images_dir=images_dir,
        text_dir=text_dir,
        instances_dir=inst_dir,
        limit=desired_count,
        save_list_path=save_index
    )
    if not index:
        logger.error("没有满足条件的样本")
        return
    logger.info(f"将处理 {len(index)} 张图像（来自：{images_dir}） | 文本目录: {text_dir} | 实例目录: {inst_dir}")

    # 单实例：避免重复占显存
    locator = create_clip_locator(cfg["locate"])
    sam     = create_sam_runner(cfg["segment"])
    packer  = create_semantic_packer(cfg["compress"])
    srdec   = create_sr_runner(cfg["recon"])

    # 目录
    pt_dir      = Path(cfg["paths"]["pt_output"]);              pt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir   = Path(cfg["paths"]["reconstruction_output"]);  recon_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(cfg["paths"]["outputs"]);                outputs_dir.mkdir(parents=True, exist_ok=True)
    save_intermediate = cfg["pipeline"].get("save_intermediate", True)

    # 运行前清空 outputs 但保留目录结构
    clean_outputs_keep_dirs(outputs_dir)

    rows = []
    for i, rec in enumerate(index, 1):
        img_path = Path(rec["image_path"])
        text     = rec.get("text", "") or ""
        if not text:
            # 虽然已保证 text 文件存在，但若首行没有 'sent'，这里仍跳过
            continue

        logging.info(f"[{i}/{len(index)}] {img_path.name} | text=\"{text}\"")

        # 读原图
        orig_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if orig_bgr is None:
            continue
        H, W = orig_bgr.shape[:2]
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # 1) 文本定位（原图坐标系）
        out_loc = locator.locate(str(img_path), text)
        if not out_loc:
            out_loc = {'bbox':[0,0,W,H], 'score':0.0}

        # 2) SAM 分割
        out_sam = sam.segment_from_bbox(str(img_path), out_loc, save_visualization=save_intermediate)
        bbox = out_sam.get("bbox") or out_loc.get("bbox") or [0,0,W,H]
        out_sam["bbox"] = bbox

        mask = out_sam["mask"]
        if mask.sum() == 0:
            # 回退：用 bbox 生成矩形 mask
            if isinstance(bbox, dict) and {"x1","y1","x2","y2"} <= set(bbox.keys()):
                x1,y1,x2,y2 = bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]
            else:
                x,y,w,h = bbox
                x1,y1,x2,y2 = x, y, x+max(1,w), y+max(1,h)
            mask = np.zeros((H, W), np.uint8)
            mask[y1:y2, x1:x2] = 1

        # 3) 固定 bpp：按原图像素数算目标比特
        target_bits = int(TARGET_BPP * W * H)
        bits_tol    = int(BPP_TOL * target_bits)

        # 4) 打包（深度学习 VGA 编解码）
        payload = packer.pack_image(
            str(img_path), mask, out_sam["bbox"],
            target_bits=target_bits, bits_tolerance=bits_tol
        )

        # 5) 重建与保存
        recon = srdec.reconstruct_image(payload)
        recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(recon_dir / f"{img_path.stem}_recon.jpg"),
                    recon_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 6) 指标
        _ssim = ssim(orig_rgb, recon, channel_axis=2, data_range=255)
        _psnr = psnr(orig_rgb, recon, data_range=255)

        rows.append({
            "image": img_path.name,
            "text": text,
            "width": W, "height": H,
            "bpp_target": TARGET_BPP,
            "bpp": payload["bpp"],
            "bits_total": payload["bits"]["total"],
            "roi_box": json.dumps(payload["roi_bbox_xyxy"]),
            "has_roi": 1,
            "ssim": _ssim,
            "psnr": _psnr
        })

        # 清理
        del orig_bgr, orig_rgb, recon, recon_bgr

    # 写 CSV
    if rows:
        out_csv = outputs_dir / "metrics_bpp_ssim_psnr.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        logging.info(f"完成：{len(rows)}/{len(index)} 张，结果写入 {out_csv}")
    else:
        logging.error("No results")

if __name__ == "__main__":
    main()
