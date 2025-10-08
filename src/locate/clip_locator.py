# src/locate/clip_locator.py
import logging, sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2, numpy as np
from utils.utils import as_pil
from typing import Union, Tuple
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from rl.utils.clip_util import get_heatmap  # 应返回 224x224 float32
except Exception as e:
    logging.warning(f"get_heatmap import failed: {e}. Using placeholder.")
    def get_heatmap(img_or_path, prompt: str) -> np.ndarray:
        # 允许传路径或 RGB 数组，便于占位
        rng = np.random.default_rng(0)
        return rng.random((224, 224), dtype=np.float32)

logger = logging.getLogger(__name__)

# ---------- helpers ----------
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

def _heatmap_to_bbox(
    hm: np.ndarray,               # 已经是目标坐标系尺寸的热图，float32，约在[0,1]
    p_high: float,
    p_low: float,
    min_area_ratio: float,
    nms_iou: float,
    top_k: int,
) -> Dict[str, Any]:
    H, W = hm.shape[:2]
    # 1) 双阈值 + 形态学
    tH = float(np.quantile(hm, p_high))
    tL = float(np.quantile(hm, p_low))
    strong = (hm >= tH).astype(np.uint8)
    weak   = (hm >= tL).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, k, 1)
    weak   = cv2.morphologyEx(weak,   cv2.MORPH_CLOSE, k, 1)

    # 2) 连通域 + 过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=4)
    strong_bool = strong.astype(bool)
    min_area = max(100, int(min_area_ratio * H * W))  # 在目标尺寸下定义最小面积阈值

    cand: List[Dict[str, Any]] = []
    for lab in range(1, num):
        comp = (labels == lab)
        if not np.any(comp & strong_bool):
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = [int(stats[lab, i]) for i in (cv2.CC_STAT_LEFT,
                                                   cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH,
                                                   cv2.CC_STAT_HEIGHT)]
        x1, y1, x2, y2 = x, y, x + w, y + h
        score = float(hm[comp].mean()) * float(np.sqrt(max(area,1)))
        cand.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'score':score})

    # 3) 兜底
    if not cand:
        thr = float(np.quantile(hm, max(p_low - 0.05, 0.0)))
        mask = (hm >= thr).astype(np.uint8)
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if num2 <= 1:
            return {'bbox':[0, 0, W, H], 'score': 0.0}
        idx = int(np.argmax(stats2[1:, cv2.CC_STAT_AREA])) + 1
        x, y, w, h = [int(stats2[idx, i]) for i in (cv2.CC_STAT_LEFT,
                                                    cv2.CC_STAT_TOP,
                                                    cv2.CC_STAT_WIDTH,
                                                    cv2.CC_STAT_HEIGHT)]
        return {'bbox':[x, y, w, h], 'score': float(hm[labels2==idx].mean())}

    # 4) NMS + 取最佳
    cand.sort(key=lambda d: d['score'], reverse=True)
    cand = cand[:top_k]
    kept: List[Dict[str, Any]] = []
    while cand:
        a = cand.pop(0); kept.append(a)
        cand = [b for b in cand if _iou_xyxy(a, b) < nms_iou]

    best = max(kept, key=lambda d: d['score'])
    return {'bbox':[best['x1'], best['y1'], best['x2']-best['x1'], best['y2']-best['y1']],
            'score': best['score']}

# ---------- main class ----------
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
        self.return_heatmap_cfg = bool(config.get('return_heatmap', False))

    # 兼容旧接口：默认在“原图尺寸”坐标系返回 bbox

    def locate(
        self,
        image_or_array: Union[str, np.ndarray, Image.Image],
        text_prompt: str,
        *,
        out_size: Tuple[int,int] | None = None,
        return_heatmap: bool | None = None,
        array_mode: str = "rgb",  # 如果传的是 OpenCV 的 BGR，就写 "bgr"
    ):
        # 统一为 PIL，拿尺寸
        pil = as_pil(image_or_array, array_mode=array_mode)
        W, H = pil.size
        tgtW, tgtH = (W, H) if out_size is None else (int(out_size[0]), int(out_size[1]))

        # 直接把 PIL/ndarray/路径喂给 get_heatmap（之前已改造为三者通吃）
        hm224 = get_heatmap(pil, text_prompt, array_mode=array_mode)
        hm = cv2.resize(hm224.astype(np.float32), (tgtW, tgtH), interpolation=cv2.INTER_LINEAR)
        hm = _nan_safe01(hm)

        ret = _heatmap_to_bbox(hm, self.p_high, self.p_low, self.min_area_ratio, self.nms_iou, self.top_k)
        if self.return_heatmap_cfg if return_heatmap is None else return_heatmap:
            ret["heatmap"] = hm
        return ret

    # # 新接口：直接吃 RGB 数组；默认在该数组的尺寸坐标系返回 bbox
    # def locate_array(
    #     self,
    #     rgb: np.ndarray,     # HxWx3, uint8
    #     text_prompt: str,
    #     *,
    #     out_size: Optional[Tuple[int,int]] = None,          # None -> 用 rgb.shape
    #     return_heatmap: Optional[bool] = None
    # ) -> Optional[Dict[str, Any]]:
    #     if rgb is None or rgb.ndim != 3:
    #         logger.error("locate_array: invalid rgb")
    #         return None
    #     H, W = rgb.shape[:2]
    #     tgtW, tgtH = (W, H) if out_size is None else (int(out_size[0]), int(out_size[1]))

    #     # 你的 get_heatmap 若仅支持路径，这里也可以先写临时文件；假设它支持 numpy 输入更佳
    #     hm224 = get_heatmap(rgb, text_prompt)
    #     if hm224 is None or hm224.ndim != 2:
    #         logger.error("heatmap invalid")
    #         return None

    #     hm = cv2.resize(hm224.astype(np.float32), (tgtW, tgtH), interpolation=cv2.INTER_LINEAR)
    #     hm = _nan_safe01(hm)

    #     ret = _heatmap_to_bbox(
    #         hm, self.p_high, self.p_low, self.min_area_ratio, self.nms_iou, self.top_k
    #     )
    #     rh = self.return_heatmap_cfg if return_heatmap is None else return_heatmap
    #     if rh:
    #         ret['heatmap'] = hm
    #     return ret

def create_clip_locator(config: Dict[str, Any]) -> CLIPLocator:
    return CLIPLocator(config)

# 简单自测
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    cfg = {
        'p_high': 0.92, 'p_low': 0.78,
        'min_area_ratio': 0.001,
        'nms_iou': 0.5, 'top_k': 5,
        'return_heatmap': True,
    }
    locator = create_clip_locator(cfg)
    img = "data/images/test.jpg"
    if Path(img).exists():
        out1 = locator.locate(img, "red apple")                               # 原图坐标
        out2 = locator.locate(img, "red apple", out_size=(512,512))           # 指定坐标系
        rgb = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        out3 = locator.locate_array(rgb, "red apple", out_size=(512,512))     # 数组接口
        print(out1.keys(), out2['bbox'], out3['bbox'], (out3['heatmap'].shape if 'heatmap' in out3 else None))
    else:
        print("Test image not found.")
