from models.clip_patch_selector import ClipPatchSelector
from PIL import Image, ImageDraw
import numpy as np
import os

def visualize_mask_from_clippatchselector(
    image_path: str,
    noun: str,
    save_path: str = "masked_noun_output.png",
    mask_color=(0, 0, 0, 180),
    grid_size=(7, 7)
):
    """
    高亮显示由 CLIPPatchSelector.get_mask_by_noun() 生成的 mask 区域
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    image_rgba = image.convert("RGBA")
    selector = ClipPatchSelector(device="cuda")
    mask, _ = selector.get_mask_by_noun(image, noun)  # mask shape: [N]
    patch_h, patch_w = grid_size
    assert mask.shape[0] == patch_h * patch_w, "Mask shape does not match patch grid"

    # 创建遮罩
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = image_rgba.size
    unit_w, unit_h = img_w // patch_w, img_h // patch_h

    for i in range(patch_h):
        for j in range(patch_w):
            idx = i * patch_w + j
            if not mask[idx]:
                x0, y0 = j * unit_w, i * unit_h
                box = (x0, y0, x0 + unit_w, y0 + unit_h)
                draw.rectangle(box, fill=mask_color)

    result = Image.alpha_composite(image_rgba, overlay).convert("RGB")
    result.save(save_path)
    print(f"✅ Masked image saved to: {save_path}")
