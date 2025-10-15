from pathlib import Path
import shutil

# ---- 新增：把路径/ndarray/PIL 统一为 PIL.Image(RGB) ----
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image

import json

def _xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + max(1.0, w))), int(round(y + max(1.0, h)))
    return [x1, y1, x2, y2]

def _clamp_xyxy(b, W, H):
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(x2, W)); y2 = max(y1+1, min(y2, H))
    return [x1, y1, x2, y2]

def _map_bbox_resize_xyxy(b, from_size, to_size):
    """原图→(W,H) 直接缩放的情形（非 letterbox）"""
    x1,y1,x2,y2 = map(float, b)
    W0,H0 = from_size; W,H = to_size
    sx, sy = W / max(1,W0), H / max(1,H0)
    return _clamp_xyxy([int(round(x1*sx)), int(round(y1*sy)),
                        int(round(x2*sx)), int(round(y2*sy))], W, H)

def _read_bbox_from_instances(inst_path: str, target_ann_id: int | None):
    """
    从 instances/<image_id>.txt 里找出 ann_id==target_ann_id 的记录；
    若 target_ann_id 为空或找不到，则退而取 area 最大的那条（iscrowd==0 优先）。
    返回: (bbox_xywh(list[float]) | None, meta_dict | None)
    """
    best = None
    best_area = -1.0
    with open(inst_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "bbox" not in obj: continue
            if target_ann_id is not None and int(obj.get("ann_id", -1)) == int(target_ann_id):
                return obj["bbox"], obj
            # 备选：挑 area 最大且非 crowd
            area = float(obj.get("area", 0.0))
            if int(obj.get("iscrowd", 0)) != 0:
                area = area * 0.5  # 轻微惩罚 crowd
            if area > best_area:
                best_area = area
                best = (obj["bbox"], obj)
    return best if best is not None else (None, None)


def get_inst_bbox(
    inst_path: str | Path,
    target_id: int | None,
    ori_size: Tuple[int, int] | None = None,   # (W0, H0)
    out_size: Tuple[int, int] | None = None,   # (W, H)
) -> Optional[list[int]]:
    """
    返回在 out_size 尺寸下的 bbox (x1,y1,x2,y2) ，整型像素坐标，已 clamp ≥1 像素宽高。
    - inst 文件里读取到的是 xywh（像素，原图坐标系）
    - 如果不给 out_size，则返回原图坐标系的 xyxy
    - 约定所有尺寸元组统一为 (W, H)
    """
    inst_path = Path(inst_path) if inst_path is not None else None
    if not inst_path or not inst_path.exists():
        return None

    bbox_xywh, meta = _read_bbox_from_instances(inst_path, target_id)
    if bbox_xywh is None:
        return None

    # 1) 原图坐标：xywh -> xyxy
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1o = int(round(x))
    y1o = int(round(y))
    x2o = int(round(x + max(1.0, w)))
    y2o = int(round(y + max(1.0, h)))

    # 2) 在原图范围内 clamp（若提供 ori_size）
    if ori_size is not None:
        W0, H0 = int(ori_size[0]), int(ori_size[1])  # (W0,H0)
        x1o = max(0, min(x1o, W0 - 1))
        y1o = max(0, min(y1o, H0 - 1))
        x2o = max(x1o + 1, min(x2o, W0))
        y2o = max(y1o + 1, min(y2o, H0))

    # 3) 若不给 out_size，直接返回原图坐标
    if out_size is None or ori_size is None or (out_size == ori_size):
        return [x1o, y1o, x2o, y2o]

    # 4) 映射到 out_size (W,H)
    W0, H0 = int(ori_size[0]), int(ori_size[1])
    W,  H  = int(out_size[0]), int(out_size[1])
    sx = W / max(1, W0)
    sy = H / max(1, H0)

    x1 = int(round(x1o * sx))
    y1 = int(round(y1o * sy))
    x2 = int(round(x2o * sx))
    y2 = int(round(y2o * sy))

    # 5) 在目标尺寸范围内 clamp，保证至少 1 像素
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    return [x1, y1, x2, y2]


def as_pil(img_input: Union[str, np.ndarray, Image.Image],
            array_mode: str = "rgb") -> Image.Image:
    """
    接受路径/ndarray/PIL，统一返回 PIL.Image(RGB)。
    - ndarray: 默认认为是 RGB（array_mode='rgb'）；如果你传的是 BGR，可改 array_mode='bgr'
    - float 数组会被视为 0..1，自动放缩到 0..255
    """
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")

    if isinstance(img_input, str):
        return Image.open(img_input).convert("RGB")

    if isinstance(img_input, np.ndarray):
        arr = img_input
        if arr.ndim == 2:  # 灰度 → 3通道
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            # 认为是 0..1 或 0..255 的浮点数据
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0) / 255.0
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        if array_mode.lower() == "bgr":
            arr = arr[..., ::-1]
        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported img_input type: {type(img_input)}")

def clean_outputs_keep_dirs(root="outputs"):
    root = Path(root).resolve()
    # 安全保护，避免误删
    assert root.name == "outputs", f"Refuse to clean non-outputs path: {root}"
    root.mkdir(parents=True, exist_ok=True)

    for sub in root.iterdir():
        if not sub.is_dir():
            # 如果 outputs 下还有零散文件，直接删掉
            try:
                sub.unlink()
            except FileNotFoundError:
                pass
            continue

        # 递归清空子目录里的所有内容，但保留子目录本身
        for child in sub.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass

        # 如果你希望顶层目录一定存在（比如 logs/payloads/…），确保一下
        sub.mkdir(parents=True, exist_ok=True)

