from PIL import Image, ImageDraw

def mask_image_by_patch(image_path, output_path, cols, rows, mask_indices):
    """
    将图像划分为 cols × rows 个 patch，并在指定 patch 上添加黑色 mask。

    参数:
    - image_path: 输入图像路径（str）
    - output_path: 输出图像路径（str）
    - cols: 横向划分的 patch 数（int）
    - rows: 纵向划分的 patch 数（int）
    - mask_indices: 要加 mask 的 patch 坐标列表，如 [(1,2), (3,5)]
    """
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    patch_width = img_width // cols
    patch_height = img_height // rows

    draw = ImageDraw.Draw(img)
    for row, col in mask_indices:
        x0 = col * patch_width
        y0 = row * patch_height
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

    img.save(output_path)
    print(f"Masked image saved to: {output_path}")

# 示例使用方式：
image_path = "data/lions.jpeg"                # 你的原图路径
output_path = "data/lion_masked.png"        # 保存路径
cols, rows = 10, 5                    # 划分为 20 列 10 行
mask_indices = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9),
                 (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9),
                (4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 8), (4, 9),
                (2,0), (2,1), (2,2), (2,3), (2,8),(2,9), (2,6),
                (3, 3), (3,8), (3,6),(3,7),(3,9)]  # 黑色遮挡位置（行, 列），从 0 开始

mask_image_by_patch(image_path, output_path, cols, rows, mask_indices)
