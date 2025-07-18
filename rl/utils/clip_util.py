# utils/clip_util.py
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from functools import lru_cache
from transformers import CLIPProcessor, CLIPModel

# ------------------ lazy singleton ------------------
@lru_cache(maxsize=1)
def _load_clip():
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval().cuda()
    return model, processor

# ------------------ main API ------------------------
@torch.no_grad()
def get_heatmap(img_path: str, prompt: str) -> np.ndarray:
    """
    Returns a 224×224 float32 heat-map in [0,1].
    """
    model, processor = _load_clip()

    # 1. preprocess
    pil_img = Image.open(img_path).convert("RGB")
    inputs  = processor(text=[prompt],
                        images=pil_img,
                        return_tensors="pt",
                        padding=True).to("cuda")

    # 2. forward
    outputs = model(**inputs, output_hidden_states=True)
    img_feat = outputs.image_embeds           # (1,512)
    txt_feat = outputs.text_embeds            # (1,512)

    # 3. patch-level token: last hidden state (includes CLS + 49/196 patches)
    # for ViT-B/32: (1, 1+49, 768) -> use token[1:]
    patch_tokens = outputs.vision_model_output.last_hidden_state[:, 1:, :]  # (1,49,768)
    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)      # unit norm
    txt_feat     = torch.nn.functional.normalize(txt_feat, dim=-1)          # (1,512)

    # 4. cosine sim (batch=1)
    sim = torch.matmul(
        patch_tokens @ model.visual_projection.T,   # (1,49,512)
        txt_feat.unsqueeze(-1)                      # (1,512,1)
    ).squeeze(-1)                                   # (1,49)

    sim = sim.reshape(1, 7, 7)   # ViT-B/32 -> 7×7 grid
    sim = torch.nn.functional.interpolate(
        sim, size=(224,224), mode="bicubic", align_corners=False
    ).clamp(min=0)

    # 5. normalize 0-1
    sim -= sim.min()
    if sim.max() > 0:
        sim /= sim.max()

    return sim.squeeze(0).cpu().numpy().astype(np.float32)
