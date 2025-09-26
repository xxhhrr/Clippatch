import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import os

def compute_patch_grid(image_size, num_rows, num_cols):
    W, H = image_size
    patch_w = W // num_cols
    patch_h = H // num_rows
    coords = []

    for i in range(num_rows):
        for j in range(num_cols):
            x1 = j * patch_w
            y1 = i * patch_h
            x2 = (j + 1) * patch_w if j < num_cols - 1 else W
            y2 = (i + 1) * patch_h if i < num_rows - 1 else H
            coords.append((x1, y1, x2, y2))
    return coords, patch_w, patch_h, num_rows, num_cols

def mask_unselected_patches(img_pil, text, num_rows=7, num_cols=7, top_k=5, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = img_pil.convert("RGB")
    W, H = img.size
    transform = T.Compose([T.Resize((H, W)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 1. 自动网格划分
    coords, patch_w, patch_h, nrows, ncols = compute_patch_grid((W, H), num_rows, num_cols)

    # 2. 分 patch
    patches = []
    for x1, y1, x2, y2 in coords:
        patch = img_tensor[:, :, y1:y2, x1:x2]
        patch = torch.nn.functional.interpolate(patch, size=(224, 224), mode='bilinear')
        patches.append(patch)
    patches = torch.cat(patches, dim=0) 
    # 3. CLIP 获取 embedding
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    with torch.no_grad():
        patch_embeddings = clip_model.get_image_features(patches)
        text_inputs = processor(text=[text], return_tensors="pt").to(device)
        text_embedding = clip_model.get_text_features(**text_inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        patch_embeddings = patch_embeddings / patch_embeddings.norm(dim=-1, keepdim=True)
        scores = (patch_embeddings @ text_embedding.T).squeeze()

    scores_np = scores.cpu().numpy()
    topk_indices = torch.topk(scores, top_k).indices.cpu().numpy().tolist()

    # 4. 绘图
    draw = Image.fromarray(np.array(img)).convert("RGBA")
    overlay = Image.new("RGBA", draw.size, (0, 0, 0, 0))
    grid = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20)

    for idx, (x1, y1, x2, y2) in enumerate(coords):
        if idx not in topk_indices:
            for x in range(x1, x2):
                for y in range(y1, y2):
                    overlay.putpixel((x, y), (255, 0, 0, 60))  # 半透明红遮罩

        # 画绿色网格线
        grid.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 150), width=2)

        # 写每个 patch 的相似度得分
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        score_text = f"{scores_np[idx]:.2f}"
        grid.text((center_x - 20, center_y - 10), score_text, font=font, fill=(255, 255, 255, 255))

    masked_result = Image.alpha_composite(draw, overlay)
    return masked_result

def split_image(image_path, rows, cols, output_dir="patches"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = Image.open(image_path)
    width, height = image.size
    print(f"Original image size: {width} x {height}")

    # 每个 patch 的宽高
    patch_width = width // cols
    patch_height = height // rows

    # 划分并保存
    count = 0
    for row in range(rows):
        for col in range(cols):
            left = col * patch_width
            upper = row * patch_height
            right = (col + 1) * patch_width
            lower = (row + 1) * patch_height

            patch = image.crop((left, upper, right, lower))
            patch_path = os.path.join(output_dir, f"patch_{row}_{col}.png")
            patch.save(patch_path)
            count += 1
    print(f"✅ Saved {count} patches to {output_dir}")

# === 示例运行 ===
if __name__ == "__main__":
    #img_path = "data/images/mscoco/train2014/COCO_train2014_000000084415.jpg"
    img_path = "data/lions.jpeg"
    text = "lions"
    # img_pil = Image.open(img_path)
    # result = mask_unselected_patches(img_pil, text, num_rows=2, num_cols=1, top_k=1)
    # result.show()
    split_image(img_path, 2, 1, "data/")