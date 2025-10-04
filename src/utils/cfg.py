# src/utils/cfg.py
from pathlib import Path
from typing import Dict, Any
import yaml, torch

REQUIRED_PATHS = ["paths.images", "paths.outputs"]
MAKE_DIRS = [
    "paths.outputs", "paths.logs", "paths.pt_output",
    "paths.sam_output", "paths.compress_output", "paths.reconstruction_output"
]

ALIASES = {
    ("segment", "checkpoint_path"): ("segment", "checkpoint"),
}

def _get(d: Dict[str, Any], dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _ensure_dirs(cfg: Dict[str, Any]):
    for dotted in MAKE_DIRS:
        p = _get(cfg, dotted)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)

def _apply_aliases(cfg: Dict[str, Any]):
    for (src_ns, src_key), (dst_ns, dst_key) in ALIASES.items():
        src = cfg.get(src_ns, {})
        dst = cfg.setdefault(dst_ns, {})
        if src_key in src and dst_key not in dst:
            dst[dst_key] = src[src_key]

def _resolve_device(d: str) -> str:
    if not d or d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for dotted in REQUIRED_PATHS:
        if _get(cfg, dotted) is None:
            raise ValueError(f"Missing required config key: {dotted}")

    sys_dev = _get(cfg, "system.device", "auto")
    for ns in ("segment", "recon", "eval"):
        dev = _get(cfg, f"{ns}.device", None)
        cfg.setdefault(ns, {})
        cfg[ns]["device"] = _resolve_device(dev or sys_dev)
    cfg["system"]["device"] = _resolve_device(sys_dev)

    _apply_aliases(cfg)
    _ensure_dirs(cfg)
    return cfg
