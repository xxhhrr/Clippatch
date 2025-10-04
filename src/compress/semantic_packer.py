# src/compress/semantic_packer.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Union
from pathlib import Path
import cv2, numpy as np, json

def _bbox_to_xyxy(bbox, W, H):
    if isinstance(bbox, dict) and "x1" in bbox:
        x1,y1,x2,y2 = int(bbox["x1"]),int(bbox["y1"]),int(bbox["x2"]),int(bbox["y2"])
    elif isinstance(bbox, dict) and "bbox" in bbox:
        x,y,w,h = [int(round(v)) for v in bbox["bbox"]]
        x1,y1,x2,y2 = x,y,x+max(w,1),y+max(h,1)
    else:
        x,y,w,h = [int(round(v)) for v in bbox]
        x1,y1,x2,y2 = x,y,x+max(w,1),y+max(h,1)
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(1, min(x2, W));   y2 = max(1, min(y2, H))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    return x1,y1,x2,y2

class SemanticPacker:
    def __init__(self, config: Dict[str, Any]):
        self.lr_scale = int(config.get('lr_scale', 4))
        self.lr_quality = int(config.get('lr_quality', 50))
        self.roi_quality = int(config.get('roi_quality', 95))
        self.margin_ratio = float(config.get('margin_ratio', 0.10))
        outdir = config.get('output_dir') or config.get('paths.compress_output') or 'outputs/payloads'
        self.output_dir = Path(outdir); self.output_dir.mkdir(parents=True, exist_ok=True)

    def pack_image(self, image_path: str, mask: np.ndarray, bbox) -> Dict[str, Any]:
        img = cv2.imread(str(image_path))
        if img is None: raise ValueError(f"Cannot load image: {image_path}")
        H,W = img.shape[:2]

        m = (mask > 0).astype(np.uint8)

        # 1) LR 模板
        lr = self._create_lr_template(img)
        lr_bytes = self._encode_jpeg(lr, quality=self.lr_quality)

        # 2) 扩展 bbox，裁剪 ROI 彩图 & 裁剪 mask
        x1,y1,x2,y2 = _bbox_to_xyxy(bbox, W, H)
        dx = int((x2-x1) * self.margin_ratio); dy = int((y2-y1) * self.margin_ratio)
        x1e, y1e = max(0, x1 - dx), max(0, y1 - dy)
        x2e, y2e = min(W, x2 + dx), min(H, y2 + dy)

        roi_patch = cv2.cvtColor(img[y1e:y2e, x1e:x2e], cv2.COLOR_BGR2RGB)  # 存 PNG/JPG 不重要，RGB更直观
        roi_mask  = m[y1e:y2e, x1e:x2e] * 255

        roi_bytes  = self._encode_jpeg(cv2.cvtColor(roi_patch, cv2.COLOR_RGB2BGR), quality=self.roi_quality)
        mask_bytes = self._encode_png(roi_mask)

        total_bits = 8 * (len(lr_bytes) + len(roi_bytes) + len(mask_bytes))
        bpp = total_bits / (H * W)

        payload = {
            "lr_template": lr_bytes,
            "roi_patch": roi_bytes,
            "roi_mask": mask_bytes,
            "roi_bbox_xyxy": [int(x1e), int(y1e), int(x2e), int(y2e)],
            "original_shape": [int(H), int(W)],
            "lr_scale": self.lr_scale,
            "bits": {"lr":len(lr_bytes)*8,"roi":len(roi_bytes)*8,"mask":len(mask_bytes)*8,"total":total_bits},
            "bpp": bpp
        }
        self._save_payload(image_path, payload)
        return payload

    def _create_lr_template(self, image: np.ndarray) -> np.ndarray:
        H,W = image.shape[:2]
        lr_h, lr_w = max(1,H//self.lr_scale), max(1,W//self.lr_scale)
        return cv2.resize(image, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode_jpeg(image_bgr: np.ndarray, quality: int = 50) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode('.jpg', image_bgr, params)
        if not ok: raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    @staticmethod
    def _encode_png(image_gray_uint8: np.ndarray) -> bytes:
        ok, buf = cv2.imencode('.png', image_gray_uint8)
        if not ok: raise RuntimeError("png encode failed")
        return buf.tobytes()

    def _save_payload(self, image_path: str, payload: Dict[str, Any]) -> None:
        name = Path(image_path).stem
        # 原始字节落盘（便于真实“发包”模拟）
        (self.output_dir / f"{name}_lr.jpg").write_bytes(payload["lr_template"])
        (self.output_dir / f"{name}_roi.jpg").write_bytes(payload["roi_patch"])
        (self.output_dir / f"{name}_mask.png").write_bytes(payload["roi_mask"])
        # 元数据（不存大字节，避免二次压缩干扰）
        meta = {k: payload[k] for k in ("roi_bbox_xyxy","original_shape","lr_scale","bits","bpp")}
        (self.output_dir / f"{name}_payload_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def create_semantic_packer(config): return SemanticPacker(config)
