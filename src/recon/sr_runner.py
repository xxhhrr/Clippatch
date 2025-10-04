# src/recon/sr_runner.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import cv2, numpy as np, torch

def _resolve_device(d: str) -> str:
    return "cuda" if (d in (None,"auto") and torch.cuda.is_available()) else (d or "cpu")

def _decode_color(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None: raise RuntimeError("decode failed")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _decode_gray(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if im is None: raise RuntimeError("decode failed")
    return im

def _feather(mask01: np.ndarray, ksize: int=5) -> np.ndarray:
    if ksize % 2 == 0: ksize += 1
    blur = cv2.GaussianBlur((mask01*255).astype(np.uint8), (ksize,ksize), 0)
    return (blur.astype(np.float32)/255.0).clip(0,1)

class SRRunner:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(_resolve_device(config.get("device","auto")))
        self.model_type = str(config.get("sr_model","bicubic")).lower()
        self.feather = int(config.get("feather",5))
        self.align_to_bbox = bool(config.get("align_to_bbox", True))
        self._sr_model = self._load_sr_model(self.model_type)

    def _load_sr_model(self, model_type: str):
        if model_type not in ("bicubic","swinir","realesrgan"): model_type = "bicubic"
        if model_type == "bicubic": return None
        print(f"[SRRunner] {model_type} not implemented yet, fallback to bicubic.")
        return None

    def _upsample_to(self, lr_rgb: np.ndarray, size_wh: Tuple[int,int]) -> np.ndarray:
        W,H = size_wh
        if self._sr_model is None:
            return cv2.resize(lr_rgb, (W,H), interpolation=cv2.INTER_CUBIC)
        # TODO: 使用真实 SR 模型
        return cv2.resize(lr_rgb, (W,H), interpolation=cv2.INTER_CUBIC)

    def reconstruct_image(self, payload: Dict[str, Any]) -> np.ndarray:
        # 新协议：lr_template / roi_patch / roi_mask / roi_bbox_xyxy / original_shape
        H, W = map(int, payload["original_shape"])
        lr = _decode_color(payload["lr_template"])
        sr = self._upsample_to(lr, (W, H))
        roi = _decode_color(payload["roi_patch"])
        m_png = _decode_gray(payload["roi_mask"])
        m01 = (m_png > 0).astype(np.uint8)
        x1,y1,x2,y2 = map(int, payload["roi_bbox_xyxy"])
        ph, pw = roi.shape[:2]
        if (y2-y1, x2-x1) != (ph, pw):
            roi = cv2.resize(roi, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
            m01 = cv2.resize(m01, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        out = sr.copy()
        alpha = _feather(m01, ksize=max(3, self.feather))
        alpha3 = np.repeat(alpha[...,None], 3, axis=2).astype(np.float32)
        base_roi = out[y1:y2, x1:x2].astype(np.float32)
        blended = roi.astype(np.float32) * alpha3 + base_roi * (1.0 - alpha3)
        out[y1:y2, x1:x2] = blended.clip(0,255).astype(np.uint8)
        return out

def create_sr_runner(config: Dict[str, Any]) -> SRRunner:
    return SRRunner(config)
