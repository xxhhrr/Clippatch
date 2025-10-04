# src/pipelines/run_mvp_sr.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging, csv, random, numpy as np, torch, json

from src.utils.cfg import load_config
from src.locate.clip_locator import create_clip_locator
from src.segment.sam_runner import create_sam_runner
from src.compress.semantic_packer import create_semantic_packer
from src.recon.sr_runner import create_sr_runner

def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def worker(img_path: Path, text: str, cfg):
    # 每个线程独立实例（避免 SamPredictor 状态冲突）
    locator = create_clip_locator(cfg["locate"])
    sam     = create_sam_runner(cfg["segment"])
    packer  = create_semantic_packer(cfg["compress"])
    srdec   = create_sr_runner(cfg["recon"])

    out_loc = locator.locate(str(img_path), text)
    if not out_loc:
        return None
    out_sam = sam.segment_from_bbox(str(img_path), out_loc, save_visualization=False)
    if not out_sam:
        return None

    payload = packer.pack_image(str(img_path), out_sam["mask"], out_sam["bbox"])
    recon = srdec.reconstruct_image(payload)

    # 这里先记录核心指标（bpp），其余评估你可以接入 eval/metrics.py
    return {
        "image": img_path.name,
        "text": text,
        "bpp": payload["bpp"],
        "bits_total": payload["bits"]["total"],
        "roi_box": json.dumps(payload["roi_bbox_xyxy"]),
    }

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    cfg = load_config("configs/mvp.yaml")
    set_seeds(cfg["pipeline"].get("seed", 42))

    images_dir = Path(cfg["paths"]["images"])
    # 你自己的样本读取逻辑：这里示例每张图配同一句假文案
    samples = [(p, "a salient object") for p in sorted(images_dir.glob("*.jpg"))]

    out_rows = []
    with ThreadPoolExecutor(max_workers=cfg["pipeline"]["max_workers"]) as ex:
        futs = [ex.submit(worker, p, t, cfg) for p, t in samples]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r: out_rows.append(r)
            except Exception as e:
                logging.error(f"worker error: {e}")

    if out_rows:
        out_csv = Path(cfg["paths"]["outputs"]) / "metrics.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            w.writeheader(); w.writerows(out_rows)
        logging.info(f"done: {len(out_rows)} samples -> {out_csv}")
    else:
        logging.warning("no results")

if __name__ == "__main__":
    main()
