import os
import re
import json
import pickle


def export_refcoco_text_and_mask(image_dir, ref_file_path, coco_ann_json, text_dir, mask_dir):
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 加载 refs 和 coco annotations
    with open(ref_file_path, "rb") as f:
        refs = pickle.load(f)
    with open(coco_ann_json, "r") as f:
        coco_data = json.load(f)

    # 建 ann_id → ann 映射
    ann_dict = {ann["id"]: ann for ann in coco_data["annotations"]}

    # 建 image_id → [ref...] 映射
    image_ref_dict = {}
    for ref in refs:
        image_id = ref["image_id"]
        if image_id not in image_ref_dict:
            image_ref_dict[image_id] = []
        image_ref_dict[image_id].append(ref)

    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg"):
            continue
        match = re.search(r"(\d+)\.jpg", fname)
        if not match:
            continue
        image_id = int(match.group(1))
        image_id_str = f"{image_id:012d}"

        if image_id not in image_ref_dict:
            continue
        ref_list = image_ref_dict[image_id]

        # ==== 写 text 目录 ====
        text_path = os.path.join(text_dir, f"{image_id_str}.txt")
        with open(text_path, "w", encoding="utf-8") as f_text:
            for ref in ref_list:
                ann_id = ref["ann_id"]
                for s in ref["sentences"]:
                    out = {
                        "sent": s["sent"].strip(),
                        "ann_id": ann_id
                    }
                    f_text.write(json.dumps(out, ensure_ascii=False) + "\n")

        # ==== 写 instances 目录 ====
        inst_path = os.path.join(mask_dir, f"{image_id_str}.txt")
        with open(inst_path, "w", encoding="utf-8") as f_inst:
            for ref in ref_list:
                ann_id = ref["ann_id"]
                if ann_id not in ann_dict:
                    continue
                ann = ann_dict[ann_id]
                desc = {
                    "ann_id": ann_id,
                    "bbox": ann.get("bbox", []),
                    "segmentation_type": "polygon" if isinstance(ann["segmentation"], list) else "rle",
                    "area": ann.get("area", 0),
                    "iscrowd": ann.get("iscrowd", 0)
                }
                f_inst.write(json.dumps(desc) + "\n")

    print(f"\n✅ 已生成文本目录：{text_dir}")
    print(f"✅ 已生成实例目录：{mask_dir}")

if __name__ == '__main__':
    export_refcoco_text_and_mask(
    image_dir="/home/ddrhx/Documents/refer/data/images/mscoco/train2014",
    ref_file_path="/home/ddrhx/Documents/refer/data/refcoco+/refs(unc).p",
    coco_ann_json="/home/ddrhx/Documents/refer/data/refcoco+/instances.json",
    text_dir="/home/ddrhx/Documents/clip_patch_compression/data/text",
    mask_dir="/home/ddrhx/Documents/clip_patch_compression/data/instances"
)