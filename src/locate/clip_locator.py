# src/locate/clip_locator.py
import logging, sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2, numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from rl.utils.clip_util import get_heatmap  # 应返回 224x224 float32
except Exception as e:
    logging.warning(f"get_heatmap import failed: {e}. Using placeholder.")
    def get_heatmap(img_path: str, prompt: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((224, 224), dtype=np.float32)

logger = logging.getLogger(__name__)

def _nan_safe01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn: return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def _iou_xyxy(a, b) -> float:
    x1 = max(a['x1'], b['x1']); y1 = max(a['y1'], b['y1'])
    x2 = min(a['x2'], b['x2']); y2 = min(a['y2'], b['y2'])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    ub = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / max(ua + ub - inter, 1e-6)

class CLIPLocator:
    def __init__(self, config: Dict[str, Any]):
        hys = config.get('hysteresis', {})
        fil = config.get('filter', {})
        nms = config.get('nms', {})
        self.p_high = float(config.get('p_high', hys.get('p_high', 0.92)))
        self.p_low  = float(config.get('p_low',  hys.get('p_low',  0.78)))
        self.min_area_ratio = float(config.get('min_area_ratio', fil.get('min_area_ratio', 0.001)))
        self.nms_iou = float(config.get('nms_iou', nms.get('iou', 0.5)))
        self.top_k   = int(config.get('top_k',   nms.get('top_k', 5)))
        self.return_heatmap = bool(config.get('return_heatmap', False))

    def locate(self, image_path: str, text_prompt: str) -> Optional[Dict[str, Any]]:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"read image failed: {image_path}")
            return None
        H, W = img.shape[:2]

        hm224 = get_heatmap(str(image_path), text_prompt)
        hm = cv2.resize(hm224.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        hm = _nan_safe01(hm)

        tH = float(np.quantile(hm, self.p_high))
        tL = float(np.quantile(hm, self.p_low))
        strong = (hm >= tH).astype(np.uint8)
        weak   = (hm >= tL).astype(np.uint8)
        k = np.ones((3,3), np.uint8)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, k, 1)
        weak   = cv2.morphologyEx(weak,   cv2.MORPH_CLOSE, k, 1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=4)
        strong_bool = strong.astype(bool)
        min_area = max(100, int(self.min_area_ratio * H * W))

        cand: List[Dict[str, Any]] = []
        for lab in range(1, num):
            comp = (labels == lab)
            if not np.any(comp & strong_bool): continue
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area: continue
            x, y, w, h = [int(stats[lab, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            x1, y1, x2, y2 = x, y, x + w, y + h
            score = float(hm[comp].mean()) * float(np.sqrt(area))
            cand.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'score':score})

        if not cand:
            thr = float(np.quantile(hm, max(self.p_low - 0.05, 0.0)))
            mask = (hm >= thr).astype(np.uint8)
            num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            if num2 <= 1: return None
            idx = int(np.argmax(stats2[1:, cv2.CC_STAT_AREA])) + 1
            x, y, w, h = [int(stats2[idx, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                        cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            x1,y1,x2,y2 = x,y,x+w,y+h
            ret = {'bbox':[x1,y1,w,h], 'score': float(hm[labels2==idx].mean())}
            if self.return_heatmap: ret['heatmap'] = hm
            return ret

        cand.sort(key=lambda d: d['score'], reverse=True)
        cand = cand[:self.top_k]
        kept: List[Dict[str, Any]] = []
        while cand:
            a = cand.pop(0); kept.append(a)
            cand = [b for b in cand if _iou_xyxy(a, b) < self.nms_iou]

        best = max(kept, key=lambda d: d['score'])
        ret = {'bbox': [best['x1'], best['y1'], best['x2']-best['x1'], best['y2']-best['y1']],
               'score': best['score']}
        return ret

def create_clip_locator(config: Dict[str, Any]) -> CLIPLocator:
    return CLIPLocator(config)



# 简单自测
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    cfg = {
        'hysteresis': {'p_high': 0.92, 'p_low': 0.78},
        'filter': {'min_area_ratio': 0.001},
        'nms': {'iou': 0.5, 'top_k': 5},
        'return_heatmap': False,
    }
    locator = create_clip_locator(cfg)
    img = "data/images/test.jpg"
    if Path(img).exists():
        out = locator.locate(img, "red apple")
        print(out)
    else:
        print("Test image not found.")
