# -*- coding: utf-8 -*-
"""
simple_mask_visualization.py

功能概览
--------
1) 基于 CLIP 热图的“训练-free”矩形框生成：
   - 224x224 heatmap -> resize 到原图尺寸 -> 0..1 归一化
   - 滞后阈值 (p_high, p_low) 生成 strong/weak mask
   - 连通域筛选：保留与 strong 重叠的 weak 连通域
   - 组件评分：mean(heat) * sqrt(area)
   - NMS 去重，保留 top-k 再取最优
   - 失败回退：单阈值取最大连通域

2) 数据产出：
   - 批量读取 data/texts 下 <image_id>.txt（JSONL）
   - 每行应包含 ann_id 与句子 sent（作为 prompt）
   - 为每个 ann_id 产出预测框，写入 data/pt/<image_id>.txt（JSONL）
   - 行格式与 data/instances 保持一致：
     {"ann_id": 123, "bbox": [x,y,w,h], "segmentation_type": "polygon", "area": 1234.56, "iscrowd": 0}

3) 可视化（可选，默认关闭）：
   - 组合图：左侧画 Pred/GT 矩形框 + 顶部 prompt 横幅，右侧为稳健着色的热图
   - 可用于抽查算法效果；大量跑数据时建议关闭

配置
----
使用 config.yaml（与本文件在同一目录）：
data:
  images: data/images
  masks: data/instances   # GT JSONL（可不使用，但保留路径）
  texts: data/texts       # 输入样本，每图一个 <image_id>.txt
  pt:    data/pt          # 预测框输出目录

hysteresis:
  p_high: 0.92
  p_low: 0.78

filter:
  min_area_ratio: 0.001

nms:
  iou: 0.5
  top_k: 5

viz:
  colormap: 2            # cv2.COLORMAP_JET
  percentile_low: 1.0
  percentile_high: 99.0
  gamma: 1.0

依赖
----
- Python 3.8+
- numpy, opencv-python, PyYAML, Pillow
- 你自己的 utils.clip_util.get_heatmap(img_path, prompt) -> np.ndarray(224x224)

命令行使用
----------
# 生成 pt 数据集（默认最多 50,000 张；不导出可视化）
python simple_mask_visualization.py --mode pt --limit 50000

# 跳过已存在的 pt/<image_id>.txt
python simple_mask_visualization.py --mode pt --limit 50000 --skip-existing

# 抽样可视化（可选，调试用；不会写 pt）
python simple_mask_visualization.py --mode viz --num-samples 50 --save-dir mask_visualization_output

注意
----
- 本脚本不会在 pt 模式下输出 PNG；viz 模式才会生成图片。
- 如果 texts/ 中同一张图的 JSONL 有重复 ann_id，默认“保留首次出现”，可在 generate_pt_dataset() 中调整。
"""

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
import argparse


# ------------------------------------------------------------
# Logger
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


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

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


def write_jsonl_lines(path: Path, lines: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Geometry & NMS
# ------------------------------------------------------------

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
# Main tool
# ------------------------------------------------------------

class SimpleBBoxVisualizer:
    """
    Generate rectangular bboxes directly from CLIP heatmaps (training-free),
    with hysteresis thresholding + connected components + scoring + NMS.
    Provides:
      - PT dataset writer: data/pt/<image_id>.txt (JSONL per image)
      - Optional visualization for debugging
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
        self.pt_dir = cdir / cfg["data"].get("pt", "data/pt")
        self.pt_dir.mkdir(parents=True, exist_ok=True)

        # Keep viz dir for optional debugging
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colors for drawing
        self.pred_color = (255, 0, 0)  # BGR red
        self.gt_color = (0, 255, 0)    # BGR green

        # Hyperparameters (configurable)
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
    # IO utils
    # -----------------------------

    @staticmethod
    def _xyxy_to_xywh(b: List[int]) -> List[float]:
        """[x1,y1,x2,y2] -> [x,y,w,h]"""
        if b is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in b]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)]

    def _resolve_image_path(self, image_id: str) -> Optional[Path]:
        """按既有规则寻找图像文件"""
        p1 = self.images_dir / f"COCO_train2014_{image_id}.jpg"
        p2 = self.images_dir / f"{image_id}.jpg"
        if p1.exists():
            return p1
        if p2.exists():
            return p2
        return None

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

        hm_u8 = (hmn * 255.0 + 0.5).astype(np.uint8)
        hm_u8 = cv2.resize(hm_u8, out_wh, interpolation=cv2.INTER_CUBIC)
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
          5) NMS -> pick best -> return bbox [x1,y1,x2,y2]
        Returns (bbox[x1,y1,x2,y2], heatmap_resized_0to1).
        """
        try:
            from utils.clip_util import get_heatmap  # <- 需由你提供
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
        boxes, scores = [], []
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
    # Visualization (optional)
    # -----------------------------

    def create_debug_visualization(self, image_bgr: np.ndarray, pred_bbox: Optional[List[int]],
                                   gt_bbox: Optional[List[int]], prompt: str,
                                   heatmap01: Optional[np.ndarray]) -> np.ndarray:
        """Compose side-by-side: image with boxes + colorized original heatmap (robust)."""
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
    # Dataset loops
    # -----------------------------

    def collect_all_text_samples(self) -> List[dict]:
        """
        汇总 data/texts/*.txt 的所有行，每行补上 image_id 字段（来自文件名）。
        注意：generate_pt_dataset() 并不使用这个函数逐条输出，而是“按图聚合”写文件。
        """
        all_samples = []
        for text_file in self.texts_dir.glob("*.txt"):
            samples = self._read_json_lines(text_file)
            for s in samples:
                s["image_id"] = text_file.stem
                all_samples.append(s)
        return all_samples

    def generate_pt_dataset(self, limit_images: int = 50000, skip_existing: bool = False, dedup: str = "first") -> None:
        """
        逐图生成 data/pt/<image_id>.txt，行格式与 instances 一致。
        - limit_images: 最多处理多少张图（按文件数量截断）
        - skip_existing: 若 pt/<image_id>.txt 已存在则跳过
        - dedup: 同一图内重复 ann_id 的策略：'first'|'last'|'none'
        """
        txt_files = sorted(self.texts_dir.glob("*.txt"))
        if not txt_files:
            self.logger.warning(f"No text files found in {self.texts_dir}")
            return

        to_process = txt_files[:min(limit_images, len(txt_files))]
        self.logger.info(f"Start generating PT JSONL for {len(to_process)} images...")
        written = 0
        skipped = 0
        empty_written = 0

        for idx, text_file in enumerate(to_process, 1):
            image_id = text_file.stem
            out_path = self.pt_dir / f"{image_id}.txt"

            if skip_existing and out_path.exists():
                skipped += 1
                if idx % 100 == 0:
                    self.logger.info(f"Progress: {idx}/{len(to_process)} (skipped={skipped}, written={written}, empty={empty_written})")
                continue

            img_path = self._resolve_image_path(image_id)
            if img_path is None:
                self.logger.debug(f"[{idx}/{len(to_process)}] image not found for {image_id}, skip.")
                continue

            samples = safe_read_json_lines(text_file)
            if not samples:
                out_path.write_text("", encoding="utf-8")
                empty_written += 1
                if idx % 100 == 0:
                    self.logger.info(f"Progress: {idx}/{len(to_process)} (skipped={skipped}, written={written}, empty={empty_written})")
                continue

            # 去重策略
            if dedup in ("first", "last"):
                buf: Dict[int, dict] = {}
                iterable = samples if dedup == "last" else []
                if dedup == "first":
                    # 保留首次出现
                    seen = set()
                    ordered = []
                    for s in samples:
                        ann_id = s.get("ann_id")
                        if ann_id is None or ann_id in seen:
                            continue
                        seen.add(ann_id)
                        ordered.append(s)
                    samples = ordered
                else:
                    # 保留最后一次出现
                    for s in samples:
                        ann_id = s.get("ann_id")
                        if ann_id is None:
                            continue
                        buf[ann_id] = s
                    samples = list(buf.values())

            out_lines = []
            for s in samples:
                ann_id = s.get("ann_id")
                if ann_id is None:
                    continue
                prompt = s.get("sent", "")

                pred_xyxy, _ = self.generate_predicted_bbox_with_heatmap(img_path, prompt)
                if pred_xyxy is None:
                    continue
                xywh = self._xyxy_to_xywh(pred_xyxy)
                if xywh is None:
                    continue
                area = round(float(xywh[2] * xywh[3]), 2)

                rec = {
                    "ann_id": ann_id,
                    "bbox": xywh,  # [x, y, w, h]
                    "segmentation_type": s.get("segmentation_type", "polygon"),
                    "area": area,
                    "iscrowd": int(s.get("iscrowd", 0)),
                }
                out_lines.append(rec)

            write_jsonl_lines(out_path, out_lines)
            written += 1

            if idx % 100 == 0 or idx == len(to_process):
                self.logger.info(f"Progress: {idx}/{len(to_process)} (skipped={skipped}, written={written}, empty={empty_written})")

        self.logger.info(f"Done. Images written={written}, skipped={skipped}, empty_files={empty_written}. PT dir: {self.pt_dir}")

    # ---- 可选：抽样可视化（调试用，不写 PT） ----

    def visualize_random_samples(self, num_samples: int = 100, save_dir: Optional[Path] = None) -> None:
        """
        随机抽取若干行（跨多图），生成 side-by-side PNG。仅用于调试。
        """
        save_dir = Path(save_dir or self.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        all_samples = self.collect_all_text_samples()
        if not all_samples:
            self.logger.warning("No samples found in texts dir.")
            return

        selected = random.sample(all_samples, min(num_samples, len(all_samples)))
        ok = 0
        for i, s in enumerate(selected, 1):
            image_id = s["image_id"]
            ann_id = s.get("ann_id")
            prompt = s.get("sent", "")

            img_path = self._resolve_image_path(image_id)
            if img_path is None:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gt_bbox = self.load_ground_truth_bbox(image_id, ann_id)  # 可选，若无 GT 会为 None
            pred_bbox, heatmap = self.generate_predicted_bbox_with_heatmap(img_path, prompt)
            if pred_bbox is None:
                continue

            debug = self.create_debug_visualization(image, pred_bbox, gt_bbox, prompt, heatmap)
            out_name = f"{image_id}_{ann_id}_debug.png"
            cv2.imwrite(str(save_dir / out_name), debug)
            ok += 1

        self.logger.info(f"Visualization done. Saved {ok}/{len(selected)} images to {save_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PT bbox JSONL or visualize samples.")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--mode", type=str, choices=["pt", "viz"], default="pt",
                   help="pt: 生成 data/pt；viz: 抽样可视化（不写 pt）")
    p.add_argument("--limit", type=int, default=50000,
                   help="[pt] 最多处理多少张图（按 texts/*.txt 文件数截断）")
    p.add_argument("--skip-existing", action="store_true",
                   help="[pt] 若 data/pt/<image_id>.txt 已存在则跳过")
    p.add_argument("--dedup", type=str, choices=["first", "last", "none"], default="first",
                   help="[pt] 同一图内重复 ann_id 的处理策略")
    p.add_argument("--num-samples", type=int, default=100,
                   help="[viz] 随机抽样可视化的样本行数")
    p.add_argument("--save-dir", type=str, default="mask_visualization_output",
                   help="[viz] 可视化图片保存目录")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main():
    args = build_argparser().parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    vis = SimpleBBoxVisualizer(
        config_path=args.config,
        output_dir=args.save-dir if hasattr(args, "save-dir") else "mask_visualization_output",
        log_level=log_level,
    )

    if args.mode == "pt":
        vis.generate_pt_dataset(limit_images=args.limit,
                                skip_existing=args.skip_existing,
                                dedup=args.dedup)
    else:
        vis.visualize_random_samples(num_samples=args.num_samples,
                                     save_dir=Path(args.save_dir))


if __name__ == "__main__":
    main()
