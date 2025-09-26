from PIL import Image, ImageDraw
import numpy as np

def generate_patch_mask_from_tree(image_size, result_tree, threshold=0.3, return_np=False):
    """
    根据递归结果树，生成与原图同尺寸的二值掩码图（仅标出最小满足条件的patch）
    """
    mask = Image.new("L", image_size, 0)  # 单通道，0 表示黑色
    draw = ImageDraw.Draw(mask)
    w, h = image_size

    def visit(node):
        region = node["region"]
        score = node["score"]
        children = node.get("children", [])

        x0 = int(region[0] * w)
        y0 = int(region[1] * h)
        x1 = int(region[2] * w)
        y1 = int(region[3] * h)

        if not children and score >= threshold:
            draw.rectangle([x0, y0, x1, y1], fill=255)

        for child in children:
            visit(child)

    visit(result_tree)

    return np.array(mask) if return_np else mask

def generate_mask_from_patch_list(image_size, region_list):
    """
    输入原图尺寸和一组 patch 区域（归一化坐标），输出 mask 图（PIL.Image）
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = image_size

    for region in region_list:
        x0 = int(region[0] * w)
        y0 = int(region[1] * h)
        x1 = int(region[2] * w)
        y1 = int(region[3] * h)
        draw.rectangle([x0, y0, x1, y1], fill=255)

    return mask




