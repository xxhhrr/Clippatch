# src/data/id_selector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json, re

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_ID_PAT = re.compile(r"(\d{12})")

def list_images(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists(): return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS])

def image_id_from_name(name: str) -> Optional[str]:
    m = _ID_PAT.search(name)
    return m.group(1) if m else None

def read_first_sent_ann_id(text_file: Path) -> Tuple[str, Optional[int]]:
    """
    读取 text/<id>.txt 第一行 JSON 的 'sent' 与同层级 'ann_id'。
    - 成功：返回 (sent, ann_id)
    - 不存在/解析失败/不存在字段：返回 ("", None)
    """
    if not text_file.exists():
        return "", None

    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sent = obj.get("sent", "")
            if not isinstance(sent, str):
                sent = ""

            # 尝试把 ann_id 转成 int（允许源是 int 或字符串数字）
            ann_id= obj.get("ann_id", None)

            if sent or ann_id is not None:
                return sent, ann_id

    return "", None



def build_id_text_list(
    images_dir: Path,
    text_dir: Path,
    instances_dir: Path,
    limit: int = 20000,
    save_list_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    收集“满足条件”的样本（有 text/<id>.txt 且 instances/<id>.txt），
    按文件名排序的确定性顺序，截断到 limit，并返回：
      {
        "image_path": str,
        "image_id": "000000123456",
        "text": str,                    # 第一行 sent；可能为空串
        "text_path": str,               # text/<id>.txt
        "inst_path": str                # instances/<id>.txt
      }
    """
    images_dir   = Path(images_dir)
    text_dir     = Path(text_dir)
    instances_dir= Path(instances_dir)

    results: List[Dict[str, Any]] = []
    for img_path in list_images(images_dir):
        img_id = image_id_from_name(img_path.name)
        if not img_id:
            continue

        tfile = text_dir / f"{img_id}.txt"
        if not tfile.exists():
            continue

        ifile = instances_dir / f"{img_id}.txt"
        if not ifile.exists():
            continue
        
        text, ann_id = read_first_sent_ann_id(tfile)
        rec = {
            "image_path": str(img_path),
            "image_id": img_id,
            "text_path": str(tfile),
            "inst_path": str(ifile),
            "text": text,  # 直接配上文本
            "ann_id": ann_id,
        }
        results.append(rec)
        if limit and len(results) >= limit:
            break

    if save_list_path:
        save_list_path = Path(save_list_path)
        save_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_list_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results
