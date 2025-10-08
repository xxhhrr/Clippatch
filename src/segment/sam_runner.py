# src/segment/sam_runner.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor

def _resolve_device(d: Optional[str]) -> str:
    if d in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d

def _bbox_to_xyxy(bbox, W: int, H: int) -> Tuple[int,int,int,int]:
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
    """每个进程/线程单独实例化"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(_resolve_device(config.get("device", "auto")))
        out = config.get("output_dir") or config.get("paths.sam_output") or "outputs/sam"
        self.output_dir = Path(out); self.output_dir.mkdir(parents=True, exist_ok=True)

        ckpt = config.get("checkpoint") or config.get("checkpoint_path")
        if not ckpt:
            raise ValueError("segment.checkpoint is required")

        model_type = config.get("model_type", "vit_b")
        self.max_side = int(config.get("max_side", 1024))   # ★ 限制长边，降低显存
        self.multimask_output = bool(config.get("multimask_output", False))  # 默认单掩码更省显存

        # --- 加载 SAM 并做推理配置（显存友好） ---
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(self.device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        if self.device.type == "cuda":
            sam.half()  # ★ 半精度

        self.sam = sam
        self.predictor = SamPredictor(self.sam)
        self._last_key: Optional[str] = None  # 用于可选缓存（本实现不缓存 features）

    def _resize_long_side_to(self, img_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
        H, W = img_rgb.shape[:2]
        s = max(H, W)
        if s <= max_side:
            return img_rgb, 1.0
        scale = max_side / float(s)
        new_w, new_h = int(W * scale), int(H * scale)
        img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_small, scale

    @torch.inference_mode()
    def segment_from_bbox(self, image_path: str, bbox, save_visualization: bool=False,
                          return_logits: bool=False) -> Optional[Dict[str, Any]]:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        H0, W0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        # 1) 降分辨率（可选，默认 1024）
        img_rgb_s, scale = self._resize_long_side_to(img_rgb, self.max_side)
        Hs, Ws = img_rgb_s.shape[:2]

        # 2) bbox 同步缩放到缩小图
        x1,y1,x2,y2 = _bbox_to_xyxy(bbox, W0, H0)
        bx = np.array([x1*scale, y1*scale, x2*scale, y2*scale], dtype=np.float32)[None, :]

        # 3) 推理（autocast + 单掩码）并立刻释放 features
        use_amp = (self.device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            self.predictor.set_image(img_rgb_s)  # 这一步会缓存 features（显存大户）
            masks, scores, logits = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=bx, multimask_output=self.multimask_output
            )
        self.predictor.reset_image()  # ★ 关键：立刻释放 features，避免显存积累

        # 4) 只取一个掩码（multimask_output=False 时本身就是 1 个）
        if self.multimask_output:
            idx = int(np.argmax(scores))  # 取最高分
            m = masks[idx]
            score = float(scores[idx])
            lg = logits[idx] if return_logits else None
        else:
            m = masks[0]
            score = float(scores[0])
            lg = logits[0] if return_logits else None

        # 5) 输出 CPU numpy，避免 GPU 常驻
        m = m.astype(np.uint8)  # {0,1}
        out = {
            "mask": m,  # 缩小图尺寸下的 mask
            "score": score,
            "bbox": {"x1": int(bx[0,0]), "y1": int(bx[0,1]), "x2": int(bx[0,2]), "y2": int(bx[0,3]),
                     "w": int(bx[0,2]-bx[0,0]), "h": int(bx[0,3]-bx[0,1])},
            "image_shape": (Hs, Ws),
            "scale_from_original": scale
        }
        if save_visualization:
            self._save_vis(Path(image_path), img_rgb_s, m, out["bbox"], score)
        if return_logits and lg is not None:
            out["logits"] = np.asarray(lg, dtype=np.float32)  # 也放 CPU

        return out

    def _save_vis(self, image_path: Path, image_rgb: np.ndarray, mask: np.ndarray,
                  bbox: Dict[str,int], score: float):
        vis = image_rgb.copy()
        overlay = np.zeros_like(vis); overlay[mask>0] = (0,255,0)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        cv2.rectangle(vis,(bbox["x1"],bbox["y1"]),(bbox["x2"],bbox["y2"]),(255,0,0),2)
        cv2.putText(vis,f"Score:{score:.3f}",(bbox["x1"],max(0,bbox["y1"]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        out = self.output_dir / f"{image_path.stem}_sam_vis.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def create_sam_runner(config: Dict[str, Any]) -> SAMRunner:
    return SAMRunner(config)


# -------- minimal CLI test --------
if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--model_type", default="vit_b", type=str)
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--max_side", default=1024, type=int)
    ap.add_argument("--bbox", default="", type=str,
                    help='自定义 bbox，格式： "x y w h" 或者 JSON: {"bbox":[x,y,w,h]} / {"x1":..,"y1":..,"x2":..,"y2":..}')
    ap.add_argument("--save_vis", default=1, type=int)
    args = ap.parse_args()

    # 解析 bbox；若未给出则取居中 40% 宽高的框
    def _parse_bbox(s: str, W: int, H: int):
        if not s:
            w, h = int(W*0.4), int(H*0.4)
            x, y = (W - w)//2, (H - h)//2
            return {"bbox": [x,y,w,h]}
        try:
            if s.strip().startswith("{"):
                return json.loads(s)
            xs = [float(v) for v in s.strip().split()]
            if len(xs) == 4:
                x,y,w,h = xs
                return {"bbox": [x,y,w,h]}
        except Exception:
            pass
        raise ValueError("bbox 参数格式错误")

    # 先读图拿尺寸，便于默认 bbox
    _img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if _img is None:
        raise FileNotFoundError(args.image)
    H0, W0 = _img.shape[:2]
    bbox = _parse_bbox(args.bbox, W0, H0)

    cfg = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "device": args.device,
        "max_side": int(args.max_side),
        "output_dir": "outputs/sam_test_cli",
        "multimask_output": False
    }
    runner = create_sam_runner(cfg)
    out = runner.segment_from_bbox(args.image, bbox, save_visualization=bool(args.save_vis))
    print(json.dumps({
        "score": out["score"] if out else None,
        "bbox": out["bbox"] if out else None,
        "image_shape": out["image_shape"] if out else None,
        "scale_from_original": out.get("scale_from_original", 1.0) if out else None
    }, ensure_ascii=False, indent=2))
