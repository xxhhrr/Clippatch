import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# helper: split image into p×p patches (returns tensor & coords)
def split_patches(img_tensor, patch_size):
    B, C, H, W = img_tensor.shape
    patches = img_tensor.unfold(2, patch_size, patch_size) \
                        .unfold(3, patch_size, patch_size)   # B,C,row,col,p,p
    rows, cols = patches.size(2), patches.size(3)
    patches = patches.permute(2,3,0,1,4,5)  # row,col,B,C,p,p
    patches = patches.reshape(rows*cols, C, patch_size, patch_size)
    coords  = [(c*patch_size, r*patch_size)           # x,y
               for r in range(rows) for c in range(cols)]
    return patches, coords, rows, cols
# ------------------------------------------------------------
def clip_select_with_avgpool(img_pil, text, psize=32, top_k=10,
                             model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- preprocess full image ---
    tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)

    # --- split into small patches ---
    patches, coords, nrows, ncols = split_patches(tensor, psize)

    # resize each patch to 224×224 for CLIP
    patches = F.interpolate(patches, size=(224,224), mode="bilinear")

    # --- CLIP features ---
    clip = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    with torch.no_grad():
        patch_feat = clip.get_image_features(patches)        # (N,D)
        txt_feat   = clip.get_text_features(**processor(text=[text], return_tensors="pt").to(device))
    patch_feat = F.normalize(patch_feat, dim=-1)
    txt_feat   = F.normalize(txt_feat, dim=-1)
    sims = (patch_feat @ txt_feat.T).squeeze()               # (N,)

    # --- reshape to grid & avg-pool (3×3 neighbourhood) ---
    grid = sims.view(1, 1, nrows, ncols)                    # (1,1,H',W')
    pooled = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
    pooled_flat = pooled.view(-1)                           # (N,)

    # --- select top-k by pooled score ---
    topk_idx = torch.topk(pooled_flat, top_k).indices.cpu().tolist()

    # --- visualise: mask un-selected patches (50 % alpha) ---
    vis = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0,0,0,0))
    for idx, (x, y) in enumerate(coords):
        if idx not in topk_idx:
            for i in range(psize):
                for j in range(psize):
                    overlay.putpixel((x+i, y+j), (0,0,0,120))   # semi-transparent
    vis = Image.alpha_composite(vis, overlay)
    return vis, pooled.squeeze().cpu().numpy().reshape(nrows,ncols)

# ------------------------------------------------------------
if __name__ == "__main__":
    img = "data/images/mscoco/train2014/COCO_train2014_000000477258.jpg"  # 替换成你的图像路径
    img_pil  = Image.open(img).convert("RGB")
    text = "white teddy bear with red bow no polka dots"
    out, scoremap = clip_select_with_avgpool(img_pil, text, psize=64, top_k=5)

    out.show()

    # optional heat-map
    plt.imshow(scoremap, cmap='hot'); plt.colorbar(); plt.title("pooled similarity"); plt.show()
