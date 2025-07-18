import os
import gymnasium as gym
import numpy as np
import random
import json
from gymnasium import spaces
from utils.clip_util import get_heatmap
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
    def __init__(self, cfg, max_depth=4):
        super().__init__()
        self.cfg = cfg
        self.max_depth = max_depth
        self.image_dir = cfg["data"]["train_images"]
        self.prompt_dir = cfg["data"]["prompts"]
        self.mask_dir = cfg["data"]["masks"]

        self.action_space = spaces.Discrete(len(SPLIT_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2, 224, 224), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        img_name = random.choice([f for f in os.listdir(self.image_dir)
                                  if f.endswith((".jpg",".png"))])
        self.image_id = os.path.splitext(img_name)[0]
        self.img_path = os.path.join(self.image_dir, img_name)

        # ---- 1) 读 annotation lines，随机选一个 ann ----------
        ann_list  = _read_json_lines(os.path.join(self.mask_dir, f"{self.image_id}.txt"))
        self.ann  = random.choice(ann_list)             # dict with ann_id, bbox, ...
        self.gt   = self._bbox_to_mask(self.ann["bbox"])  # 224×224

        # ---- 2) 读 prompt lines，筛选相同 ann_id，随机挑一句 -----
        prompt_lines = _read_json_lines(os.path.join(self.prompt_dir, f"{self.image_id}.txt"))
        same_id_txts = [x["sent"] for x in prompt_lines if x["ann_id"] == self.ann["ann_id"]]
        self.prompt  = random.choice(same_id_txts) if same_id_txts else ""

        # ---- 3) CLIP 热图 + 初始化 patch 列表 --------------------
        self.heat  = get_heatmap(self.img_path, self.prompt)     # (224,224)
        self.patch_list = [np.ones_like(self.heat, dtype=np.uint8)]
        self.depth = 0
        return self._get_obs(), {}

    # ---------- util: bbox → 224×224 mask ----------------------
    def _bbox_to_mask(self, bbox, size=(224,224), orig_wh=(640,480)):
        x,y,w,h = bbox
        H,W = size
        ow, oh = orig_wh
        x1 = int(np.clip(x       /ow * W, 0, W-1))
        y1 = int(np.clip(y       /oh * H, 0, H-1))
        x2 = int(np.clip((x+w) /ow * W, 0, W-1))
        y2 = int(np.clip((y+h) /oh * H, 0, H-1))
        mask = np.zeros(size, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def _get_obs(self):
        mask = np.clip(np.sum(self.patch_list, axis=0), 0, 1)
        obs = np.stack([self.heat, mask], axis=0).astype(np.float32)
        return obs

    def step(self, action):
        split = SPLIT_ACTIONS[action]
        if split == "STOP" or self.depth >= self.max_depth:
            reward = self._final_reward()
            return self._get_obs(), reward, True, False, {}

        target_idx = np.argmin([self._patch_iou(p) for p in self.patch_list])
        patch = self.patch_list.pop(target_idx)
        h, w = patch.shape
        r, c = split
        h_step, w_step = h // r, w // c

        for i in range(r):
            for j in range(c):
                sub = np.zeros_like(patch, dtype=np.uint8)
                sub[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = \
                    patch[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                self.patch_list.append(sub)

        self.depth += 1
        reward = self._step_reward()
        return self._get_obs(), reward, False, False, {}

    def _patch_iou(self, patch):
        inter = np.logical_and(patch, self.gt).sum()
        union = np.logical_or(patch, self.gt).sum()
        return inter / (union + 1e-6)

    def _step_reward(self):
        mask_new = np.clip(np.sum(self.patch_list, 0), 0, 1)
        iou = (np.logical_and(mask_new, self.gt).sum() /
               (np.logical_or(mask_new, self.gt).sum() + 1e-6))
        area_penalty = mask_new.sum() / self.gt.size
        clip_score = (mask_new * self.heat).sum() / (self.heat.sum() + 1e-6)
        return 1.0 * iou - 0.25 * area_penalty + 0.5 * clip_score

    def _final_reward(self):
        mask_new = np.clip(np.sum(self.patch_list, 0), 0, 1)
        iou = (np.logical_and(mask_new, self.gt).sum() /
               (np.logical_or(mask_new, self.gt).sum() + 1e-6))
        return 10 * iou
