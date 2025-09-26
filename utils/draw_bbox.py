import os
import json
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import cm

def draw_refcoco_with_local_instances(image_dir, text_dir, instance_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有可用图像 ID
    all_ids = [f[:-4] for f in os.listdir(text_dir) if f.endswith(".txt")]
    selected_ids = random.sample(all_ids, min(num_samples, len(all_ids)))

    def id2color(id):
        cmap = cm.get_cmap("tab20")
        return tuple(int(x * 255) for x in cmap(id % 20)[:3])

    for image_id in selected_ids:
        image_path = os.path.join(image_dir, f"COCO_train2014_{image_id}.jpg")
        text_path = os.path.join(text_dir, f"{image_id}.txt")
        instance_path = os.path.join(instance_dir, f"{image_id}.txt")

        if not all(os.path.exists(p) for p in [image_path, text_path, instance_path]):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18)

        # 加载 instance 数据（每个 ann_id → ann_info）
        with open(instance_path, "r", encoding="utf-8") as f:
            ann_map = {}
            for line in f:
                try:
                    ann = json.loads(line.strip())
                    ann_map[ann["ann_id"]] = ann
                except:
                    continue

        # 加载文本描述，每个句子附带 ann_id
        ann_text_map = {}
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                aid = obj["ann_id"]
                sent = obj["sent"]
                if aid not in ann_text_map:
                    ann_text_map[aid] = []
                ann_text_map[aid].append(sent)

        # 遍历每个 ann_id，画框、写句子
        for aid, sents in ann_text_map.items():
            if aid not in ann_map:
                continue
            ann = ann_map[aid] if isinstance(ann_map, dict) else ann_map[aid]
            if "bbox" not in ann:
                continue
            x, y, w, h = map(int, ann["bbox"])
            color = id2color(int(aid))

            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            for i, sent in enumerate(sents[:3]):  # 最多3句
                draw.text((x + 3, y + i * 20 - 20), sent, font=font, fill=color)

        # 保存图像
        out_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
        out_path = os.path.join(output_dir, f"{image_id}_vis.jpg")
        cv2.imwrite(out_path, out_img)
        print(f"✅ saved: {out_path}")


if __name__ == '__main__':
    draw_refcoco_with_local_instances(
    image_dir="data/images/mscoco/train2014",
    text_dir="data/text",
    instance_dir="data/instances",
    output_dir="data/vis_output",
    num_samples=100
)
