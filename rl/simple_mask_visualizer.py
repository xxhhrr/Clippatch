
import os
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def setup_logger(name: str = "bbox_vis", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def safe_read_json_lines(path: Path) -> List[dict]:
    """Read JSONL with best-effort parsing and silence errors."""
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return results


def bbox_iou(b1: List[int], b2: List[int]) -> float:
    if b1 is None or b2 is None:
        return 0.0
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1 = (x12 - x11) * (y12 - y11)
    a2 = (x22 - x21) * (y22 - y21)
    union = a1 + a2 - inter
    return float(inter) / float(union) if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5, top_k: Optional[int] = None) -> List[int]:
    """Pure numpy NMS, returns indices to keep."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if top_k is not None and len(keep) >= top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


# ------------------------------------------------------------
# Main class
# ------------------------------------------------------------

class SimpleBBoxVisualizer:
    """
    Generate rectangular bboxes directly from CLIP heatmaps (training-free),
    with hysteresis thresholding + connected components + scoring + NMS.
    Provides visualization and summary report with IoU against GT.
    """

    def __init__(self, config_path: str = "config.yaml", output_dir: str = "mask_visualization_output",
                 log_level=logging.INFO):
        self.logger = setup_logger(level=log_level)

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cdir = Path(config_path).parent
        self.images_dir = cdir / cfg["data"]["images"]
        self.masks_dir = cdir / cfg["data"]["masks"]
        self.texts_dir = cdir / cfg["data"]["texts"]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colors for drawing
        self.pred_color = (255, 0, 0)  # BGR red
        self.gt_color = (0, 255, 0)    # BGR green

        # Hyperparameters (can be moved into YAML)
        self.p_high = float(cfg.get("hysteresis", {}).get("p_high", 0.92))
        self.p_low = float(cfg.get("hysteresis", {}).get("p_low", 0.78))
        self.min_area_ratio = float(cfg.get("filter", {}).get("min_area_ratio", 0.001))
        self.nms_iou = float(cfg.get("nms", {}).get("iou", 0.5))
        self.top_k = int(cfg.get("nms", {}).get("top_k", 5))
        self.colormap = int(cfg.get("viz", {}).get("colormap", cv2.COLORMAP_JET))
        self.clip_low = float(cfg.get("viz", {}).get("percentile_low", 1.0))
        self.clip_high = float(cfg.get("viz", {}).get("percentile_high", 99.0))
        self.gamma = float(cfg.get("viz", {}).get("gamma", 1.0))

    # -----------------------------
    # IO
    # -----------------------------

    def _read_json_lines(self, path: Path) -> List[dict]:
        return safe_read_json_lines(path)

    def load_ground_truth_bbox(self, image_id: str, ann_id) -> Optional[List[int]]:
        """
        Read GT bbox [x, y, w, h] from JSONL at masks_dir/{image_id}.txt
        and convert to [x1, y1, x2, y2].
        """
        mask_file = self.masks_dir / f"{image_id}.txt"
        if not mask_file.exists():
            self.logger.debug(f"Mask file not found: {mask_file}")
            return None
        rows = self._read_json_lines(mask_file)
        if not rows:
            return None
        for r in rows:
            if r.get("ann_id") == ann_id and "bbox" in r:
                x, y, w, h = r["bbox"]
                return [int(x), int(y), int(x + w), int(y + h)]
        return None

    # -----------------------------
    # Heatmap utilities
    # -----------------------------

    @staticmethod
    def _normalize01(hm: np.ndarray) -> np.ndarray:
        hm = hm.astype(np.float32)
        hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = hm.min(), hm.max()
        if vmax <= vmin:
            return np.zeros_like(hm, dtype=np.float32)
        return (hm - vmin) / (vmax - vmin)

    def _colorize_heatmap(self, hm: np.ndarray, out_wh: Tuple[int, int],
                          percentile_clip: Tuple[float, float] = None) -> np.ndarray:
        """
        Robust colorization: percentile clipping -> [0,1] -> optional gamma -> uint8 -> colormap.
        """
        hmn = hm.astype(np.float32)
        hmn = np.nan_to_num(hmn, nan=0.0, posinf=0.0, neginf=0.0)

        if percentile_clip is None:
            vmin = float(hmn.min())
            vmax = float(hmn.max())
        else:
            lo, hi = percentile_clip
            vmin = float(np.percentile(hmn, lo))
            vmax = float(np.percentile(hmn, hi))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        hmn = (hmn - vmin) / (vmax - vmin)
        hmn = np.clip(hmn, 0.0, 1.0)
        if self.gamma != 1.0:
            hmn = np.power(hmn, self.gamma)

        hmn = cv2.resize(hmn, out_wh, interpolation=cv2.INTER_CUBIC)
        hm_u8 = (hmn * 255.0 + 0.5).astype(np.uint8)
        colored = cv2.applyColorMap(hm_u8, self.colormap)
        return colored

    # -----------------------------
    # Core: bbox generation (training-free)
    # -----------------------------

    def generate_predicted_bbox_with_heatmap(self, img_path: Path, prompt: str) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
        """
        Main path:
          1) get heatmap(224x224) -> resize to image size -> normalize to [0,1]
          2) hysteresis thresholding (p_high, p_low)
          3) connected components on weak; keep components that overlap 'strong'
          4) score each component by mean heat * sqrt(area)
          5) NMS -> pick best -> return bbox
        Returns (bbox[x1,y1,x2,y2], heatmap_resized_0to1).
        """
        try:
            from utils.clip_util import get_heatmap  # user-provided
        except Exception as e:
            self.logger.error(f"Please provide utils.clip_util.get_heatmap: {e}")
            return None, None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None, None
        H, W = img_bgr.shape[:2]

        # Heatmap 224 -> image size, then 0..1
        hm224 = get_heatmap(str(img_path), prompt)
        hm = cv2.resize(hm224.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        hm = self._normalize01(hm)

        # Hysteresis thresholds
        t_high = float(np.quantile(hm, self.p_high))
        t_low = float(np.quantile(hm, self.p_low))
        strong = (hm >= t_high).astype(np.uint8)
        weak = (hm >= t_low).astype(np.uint8)

        # Morphology light clean
        k = np.ones((3, 3), np.uint8)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, k, iterations=1)
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, k, iterations=1)

        # Keep weak components that overlap strong
        num, labels = cv2.connectedComponents(weak)
        strong_bool = strong.astype(bool)
        boxes = []
        scores = []
        min_area = max(100, int(self.min_area_ratio * H * W))

        for lab in range(1, num):
            comp = (labels == lab)
            if not np.any(comp & strong_bool):
                continue
            ys, xs = np.where(comp)
            if ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w, h = x2 - x1 + 1, y2 - y1 + 1
            area = w * h
            if area < min_area:
                continue
            avg_h = float(hm[ys, xs].mean())
            score = avg_h * float(np.sqrt(area))
            boxes.append([x1, y1, x2, y2])
            scores.append(score)

        # If we have candidates: NMS + take best
        if boxes:
            b = np.array(boxes, dtype=np.float32)
            s = np.array(scores, dtype=np.float32)
            keep = nms(b, s, iou_thr=self.nms_iou, top_k=self.top_k)
            best = b[keep][0].astype(int).tolist()
            return best, hm

        # Fallback: single threshold (slightly lower) -> largest CC
        thr = float(np.quantile(hm, max(self.p_low - 0.05, 0.0)))
        mask = (hm >= thr).astype(np.uint8)
        num, labels = cv2.connectedComponents(mask)
        best_box, best_area = None, -1
        for lab in range(1, num):
            ys, xs = np.where(labels == lab)
            if ys.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area > best_area:
                best_area = area
                best_box = [x1, y1, x2, y2]
        return best_box, hm

    # -----------------------------
    # Visualization
    # -----------------------------

    def create_debug_visualization(self, image_bgr: np.ndarray, pred_bbox: Optional[List[int]],
                                   gt_bbox: Optional[List[int]], prompt: str,
                                   heatmap01: Optional[np.ndarray]) -> np.ndarray:
        """Compose side-by-side: image with boxes + colorized original heatmap (robust)."""
        # left: draw on copy
        img_bgr = image_bgr.copy()
        if pred_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in pred_bbox]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), self.pred_color, 2)
            cv2.putText(img_bgr, "Pred", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.pred_color, 2)
        if gt_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in gt_bbox]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), self.gt_color, 2)
            cv2.putText(img_bgr, "GT", (x1, max(0, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.gt_color, 2)

        # prompt banner
        if prompt:
            overlay = img_bgr.copy()
            text = f"Prompt: {prompt}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
            img_bgr = cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0.0)
            cv2.putText(img_bgr, text, (12, 12 + th - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        H, W = img_bgr.shape[:2]
        if heatmap01 is None:
            right = np.zeros_like(img_bgr)
        else:
            right = self._colorize_heatmap(heatmap01, (W, H), percentile_clip=(self.clip_low, self.clip_high))

        combined = np.hstack([img_bgr, right])
        return combined

    # -----------------------------
    # Pipeline
    # -----------------------------

    def process_single_sample(self, sample_data: dict) -> Optional[dict]:
        image_id = sample_data["image_id"]
        ann_id = sample_data["ann_id"]
        prompt = sample_data["sent"]

        # Resolve image path
        if "image_path" in sample_data:
            img_path = Path(sample_data["image_path"])
        else:
            p = self.images_dir / f"COCO_train2014_{image_id}.jpg"
            img_path = p if p.exists() else self.images_dir / f"{image_id}.jpg"
            if not img_path.exists():
                return None

        image = cv2.imread(str(img_path))
        if image is None:
            return None

        gt_bbox = self.load_ground_truth_bbox(image_id, ann_id)
        if gt_bbox is None:
            return None

        pred_bbox, heatmap = self.generate_predicted_bbox_with_heatmap(img_path, prompt)

        debug_image = self.create_debug_visualization(image, pred_bbox, gt_bbox, prompt, heatmap)
        iou = bbox_iou(pred_bbox, gt_bbox)

        return {
            "image_id": image_id,
            "ann_id": ann_id,
            "prompt": prompt,
            "result_image": debug_image,
            "pred_bbox": pred_bbox,
            "gt_bbox": gt_bbox,
            "heatmap": heatmap,
            "iou": iou,
        }

    # -----------------------------
    # Reporting
    # -----------------------------

    @staticmethod
    def _iou_level(iou: float) -> str:
        if iou >= 0.8:
            return "Excellent (≥0.8)"
        if iou >= 0.6:
            return "Good (0.6–0.8)"
        if iou >= 0.4:
            return "Fair (0.4–0.6)"
        if iou >= 0.2:
            return "Poor (0.2–0.4)"
        return "Failed (<0.2)"

    def generate_summary_report(self, results: List[dict]) -> None:
        report_path = self.output_dir / "visualization_report.txt"
        total = len(results)
        valid = [r for r in results if r.get("pred_bbox") is not None and r.get("gt_bbox") is not None]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Visualization Report ===\n\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Valid pairs: {len(valid)}\n\n")
            if valid:
                ious = [r["iou"] for r in valid]
                f.write(f"Mean IoU: {np.mean(ious):.4f}\n")
                f.write(f"Max IoU: {np.max(ious):.4f}\n")
                f.write(f"Min IoU: {np.min(ious):.4f}\n\n")
                # Buckets
                buckets = {"Excellent (≥0.8)": 0, "Good (0.6–0.8)": 0, "Fair (0.4–0.6)": 0,
                           "Poor (0.2–0.4)": 0, "Failed (<0.2)": 0}
                for x in ious:
                    buckets[self._iou_level(x)] += 1
                f.write("IoU buckets:\n")
                for k, v in buckets.items():
                    pct = 100.0 * v / len(valid)
                    f.write(f"  {k}: {v} ({pct:.1f}%)\n")
                f.write("\n")
            else:
                f.write("No valid samples.\n\n")

            f.write("=== Per-sample ===\n")
            for i, r in enumerate(results, 1):
                f.write(f"{i}. image_id={r['image_id']} ann_id={r['ann_id']} prompt={r['prompt'][:40]}...\n")
                f.write(f"   pred_bbox={r.get('pred_bbox')} gt_bbox={r.get('gt_bbox')} IoU={r.get('iou', 0.0):.4f}\n")
                f.write(f"   output={r.get('output_path')}\n\n")
        self.logger.info(f"Report written to: {report_path}")

    # -----------------------------
    # Dataset loop
    # -----------------------------

    def collect_all_samples(self) -> List[dict]:
        all_samples = []
        for text_file in self.texts_dir.glob("*.txt"):
            samples = self._read_json_lines(text_file)
            for s in samples:
                s["image_id"] = text_file.stem
                all_samples.append(s)
        return all_samples

    def generate_random_visualizations(self, num_samples: int = 1000) -> List[dict]:
        self.logger.info(f"Start generating {num_samples} visualizations...")
        all_samples = self.collect_all_samples()
        if not all_samples:
            self.logger.warning("No samples found.")
            return []
        selected = random.sample(all_samples, min(num_samples, len(all_samples)))

        results = []
        ok = 0
        for i, sample in enumerate(selected, 1):
            self.logger.info(f"[{i}/{len(selected)}] image_id={sample.get('image_id')} ann_id={sample.get('ann_id')}")
            out = self.process_single_sample(sample)
            if out is None:
                continue

            # Save side-by-side image
            out_name = f"{out['image_id']}_{out['ann_id']}_debug.png"
            out_path = self.output_dir / out_name
            cv2.imwrite(str(out_path), out["result_image"])
            out_record = {
                "image_id": out["image_id"],
                "ann_id": out["ann_id"],
                "prompt": out["prompt"],
                "output_path": str(out_path),
                "pred_bbox": out["pred_bbox"],
                "gt_bbox": out["gt_bbox"],
                "iou": out["iou"],
            }
            results.append(out_record)
            ok += 1

        self.logger.info(f"Done. Successful: {ok}/{len(selected)}. Output dir: {self.output_dir}")
        self.generate_summary_report(results)
        return results


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":
    vis = SimpleBBoxVisualizer(
        config_path="config.yaml",
        output_dir="mask_visualization_output",
        log_level=logging.INFO,
    )
    # Generate visualizations
    vis.generate_random_visualizations(num_samples=1000)
