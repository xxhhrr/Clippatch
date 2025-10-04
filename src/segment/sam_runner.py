# src/segment/sam_runner.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor

def _resolve_device(d: str) -> str:
    return "cuda" if (d in (None,"auto") and torch.cuda.is_available()) else (d or "cpu")

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

class SAMRunner:
    """每个线程/进程各自实例，勿共享"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(_resolve_device(config.get("device","auto")))
        out = config.get("output_dir") or config.get("paths.sam_output") or "outputs/sam"
        self.output_dir = Path(out); self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = config.get("checkpoint") or config.get("checkpoint_path")
        if not ckpt: raise ValueError("segment.checkpoint is required")
        model_type = config.get("model_type","vit_b")
        self.sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        self._last_key: Optional[str] = None

    @torch.inference_mode()
    def segment_from_bbox(self, image_path: str, bbox, save_visualization: bool=False,
                          return_logits: bool=False) -> Optional[Dict[str, Any]]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None: return None
        H,W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        key = f"{image_path}|{H}x{W}"
        if key != self._last_key:
            self.predictor.set_image(img_rgb)
            self._last_key = key

        x1,y1,x2,y2 = _bbox_to_xyxy(bbox, W, H)
        box = np.array([x1,y1,x2,y2], dtype=np.float32)[None,:]
        use_amp = (self.device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            masks, scores, logits = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=box, multimask_output=False)

        m = (masks[0].astype(np.uint8))  # {0,1}
        out = {
            "mask": m,
            "score": float(scores[0]),
            "bbox": {"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                     "w":int(x2-x1),"h":int(y2-y1)},
            "image_shape": (H,W)
        }
        if save_visualization:
            self._save_vis(Path(image_path), img_rgb, m, out["bbox"], out["score"])
        if return_logits:
            out["logits"] = logits[0]
        return out

    def _save_vis(self, image_path: Path, image_rgb: np.ndarray, mask: np.ndarray, bbox: Dict[str,int], score: float):
        vis = image_rgb.copy()
        overlay = np.zeros_like(vis); overlay[mask>0] = (0,255,0)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        cv2.rectangle(vis,(bbox["x1"],bbox["y1"]),(bbox["x2"],bbox["y2"]),(255,0,0),2)
        cv2.putText(vis,f"Score:{score:.3f}",(bbox["x1"],max(0,bbox["y1"]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        out = self.output_dir / f"{image_path.stem}_sam_vis.png"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def create_sam_runner(config: Dict[str, Any]) -> SAMRunner:
    return SAMRunner(config)
