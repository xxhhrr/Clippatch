# models/policy_transformer.py
import torch, torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel

class GridPolicyNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.vit   = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.txt   = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        self.cross = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.embed_patch = nn.Linear(2*224*224, 768)  # heat+mask flatten demo
        self.policy = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.value  = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, prompt_ids):
        B = obs.shape[0]
        x_patch = self.embed_patch(obs.view(B, -1))
        x_txt   = self.txt(prompt_ids).pooler_output
        fused   = self.cross(x_patch.unsqueeze(0), src_key_padding_mask=None)[0] + x_txt
        return self.policy(fused), self.value(fused)
