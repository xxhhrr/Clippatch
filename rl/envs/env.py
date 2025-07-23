import os
import gymnasium as gym
import numpy as np
import random
import json
from gymnasium import spaces
from utils.clip_util import get_heatmap, print_gpu_tensors
from PIL import Image

SPLIT_ACTIONS = [
    (1, 2), (2, 1), (2, 2),
    (1, 3), (3, 1), (3, 3),
    (1, 5), (5, 1), "STOP"
]

def _read_json_lines(path):
    with open(path, 'r') as f:
        return [json.loads(l.strip()) for l in f if l.strip()]

class ClipGridEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_dir = cfg["data"]["images"]
        self.text_dir = cfg["data"]["texts"]
        self.mask_dir = cfg["data"]["masks"]

        self.action_space = spaces.Discrete(len(SPLIT_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1, 224, 224), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        print_gpu_tensors(locals(), context="Before reset")

        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
        while True:
            img_name = random.choice(image_files)
            # Extract numeric ID from filename (e.g., ..._000000000009.jpg -> 000000000009)
            try:
                self.image_id = img_name.split('_')[-1].split('.')[0]
                # zfill to pad with zeros to 12 digits
                self.image_id = self.image_id.zfill(12)
            except (IndexError, ValueError):
                continue # Skip files with unexpected names

            text_path = os.path.join(self.text_dir, f"{self.image_id}.txt")
            mask_path = os.path.join(self.mask_dir, f"{self.image_id}.txt")

            # Check if corresponding text and mask files exist before trying to load
            if not (os.path.exists(text_path) and os.path.exists(mask_path)):
                continue

            self.img_path = os.path.join(self.image_dir, img_name)
            prompts = _read_json_lines(text_path)
            instances = _read_json_lines(mask_path)

            if not prompts or not instances:
                continue

            instance_map = {inst['ann_id']: inst for inst in instances}
            valid_prompts = [p for p in prompts if p['ann_id'] in instance_map]

            if not valid_prompts:
                continue

            self.prompt_data = random.choice(valid_prompts)
            self.ann_data = instance_map[self.prompt_data['ann_id']]
            self.prompt = self.prompt_data['sent']

            with Image.open(self.img_path) as img:
                orig_wh = img.size
            self.gt = self._bbox_to_mask(self.ann_data["bbox"], orig_wh=orig_wh)

            self.heat = get_heatmap(self.img_path, self.prompt)
            print_gpu_tensors(locals(), context="After get_heatmap in reset")
            self.patch_list = [np.ones_like(self.heat, dtype=np.uint8)]
            break

        return self._get_obs(), {}


    def _bbox_to_mask(self, bbox, size=(224,224), orig_wh=(640,480)):
        x,y,w,h = bbox
        H,W = size
        ow, oh = orig_wh
        x1 = int(np.clip(x / ow * W, 0, W-1))
        y1 = int(np.clip(y / oh * H, 0, H-1))
        x2 = int(np.clip((x+w) / ow * W, 0, W-1))
        y2 = int(np.clip((y+h) / oh * H, 0, H-1))
        mask = np.zeros(size, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def _get_obs(self):
        # The observation is the heatmap.
        # We must detach it from the computation graph and move it to the CPU.
        # self.heat is already a clean numpy array returned by get_heatmap.
        # We return a copy to prevent the observation stored in the rollout buffer
        # from being accidentally modified by external code.
        obs = self.heat.copy()
        return np.expand_dims(obs, axis=0).astype(np.float32)

    def step(self, action):
        print_gpu_tensors(locals(), context="Start of step")
        split = SPLIT_ACTIONS[action]

        # If action is STOP, calculate reward based on the whole image and terminate.
        if split == "STOP":
            self.patch_list = [np.ones_like(self.heat, dtype=np.uint8)]
            reward = self._final_reward()
            return self._get_obs(), reward, True, False, {}

        # Per user clarification, we split the *entire* image space once.
        full_patch = np.ones_like(self.heat, dtype=np.uint8)
        rows, cols = np.where(full_patch)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        r, c = split
        h_step = (y_max - y_min + 1) // r
        w_step = (x_max - x_min + 1) // c

        new_patches = []
        for i in range(r):
            for j in range(c):
                sub = np.zeros_like(full_patch, dtype=np.uint8)
                y_start, y_end = y_min + i * h_step, y_min + (i + 1) * h_step
                x_start, x_end = x_min + j * w_step, x_min + (j + 1) * w_step
                sub[y_start:y_end, x_start:x_end] = 1
                if sub.sum() > 0:
                    new_patches.append(sub)

        # The new patch list *is* the result of the single split.
        self.patch_list = new_patches
        
        # The episode ends after this single split.
        reward = self._final_reward()
        
        print_gpu_tensors(locals(), context="End of step")
        return self._get_obs(), reward, True, False, {}

    def _final_reward(self):
        if not self.patch_list:
            return 0.0

        clip_scores = []
        for patch in self.patch_list:
            # Use the patch as a mask on the heatmap to get the score
            score = (self.heat * patch).sum()
            clip_scores.append(score)

        # Find the patch with the highest CLIP score
        best_patch_idx = np.argmax(clip_scores)
        best_patch = self.patch_list[best_patch_idx]

        # Calculate IoU only for the best patch
        inter = np.logical_and(best_patch, self.gt).sum()
        union = np.logical_or(best_patch, self.gt).sum()
        iou = inter / (union + 1e-6)

        # The reward is the IoU of the best patch.
        # No penalty for the number of patches, as we only select one.
        return iou
