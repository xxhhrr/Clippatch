import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ========== Step 1: 设置模型路径和类型 ==========
sam_checkpoint = "segment-anything/weights/sam_vit_h_4b8939.pth"  # 修改为你自己的路径
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Step 2: 加载 SAM 模型 ==========
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# ========== Step 3: 创建自动分割器 ==========
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# ========== Step 4: 读取图像 ==========
image_path = "data/worldcup.jpg"  # 替换为你自己的图像路径
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ========== Step 5: 计算分割耗时 ==========
start_time = time()
masks = mask_generator.generate(image)
end_time = time()
print(f"Total masks generated: {len(masks)}")
print(f"Mask generation time: {end_time - start_time:.2f} seconds")

# ========== Step 6: 显示函数 ==========
def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# ========== Step 7: 可视化结果 ==========
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title("Segment Anything Automatic Masks")
plt.show()
