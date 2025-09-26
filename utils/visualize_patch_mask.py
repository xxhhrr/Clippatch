# utils/visualize_patch_mask.py

from PIL import Image, ImageDraw
import numpy as np

def visualize_patch_mask(image: Image.Image, scores, grid_size=(10, 10), topk=10, save_path="output.png"):
    image = image.resize((512, 512)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    pw, ph = w // grid_size[0], h // grid_size[1]

    topk_patches = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]
    for (i, j), score in topk_patches:
        box = (j * pw, i * ph, (j+1) * pw, (i+1) * ph)
        draw.rectangle(box, fill=(255, 0, 0, 120))

    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.save(save_path)
    print(f"✅ Saved to {save_path}")
