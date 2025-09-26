import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# from vit_explain import GradCAM

def visualize_gradcam_map(image, cam, image_size):
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = torch.nn.functional.interpolate(
        torch.tensor(cam)[None, None], size=image_size, mode='bilinear', align_corners=False
    )[0, 0].numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title("Grad-CAM Heatmap")
    plt.show()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=[text_prompt], return_tensors="pt", padding=True).to(device)

    gradcam = GradCAM(model=model, target_layer="vision_model.encoder.layers.11.output.LayerNorm", reshape_transform=None)
    model.eval()

    def get_score(output):
        image_features = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
        text_features = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T)[0][0]

    with gradcam.activate_and_compute_gradients(inputs, get_score) as cam:
        visualize_gradcam_map(image, cam.cpu().numpy(), image.size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--text", required=True, help="Text prompt")
    args = parser.parse_args()
    main(args.image, args.text)
