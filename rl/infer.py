import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw
from envs.env import ClipGridEnv, SPLIT_ACTIONS
from models.rl_agent import CustomActorCriticPolicy
from stable_baselines3 import PPO
from utils.clip_util import get_heatmap

cfg = yaml.safe_load(open("config.yaml"))

# 加载模型
model = PPO.load(cfg["model_save_path"])

def draw_patches(image, patch_list):
    draw = ImageDraw.Draw(image)
    for patch in patch_list:
        rows, cols = np.where(patch)
        if len(rows) > 0 and len(cols) > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    return image

def infer(img_path, prompt):
    # 创建一个临时的 env 用于推理
    env = ClipGridEnv(cfg)
    obs, _ = env.reset()
    env.img_path = img_path
    env.prompt = prompt
    env.heat = get_heatmap(img_path, prompt)
    obs = env._get_obs()

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action.item())

    # 可视化
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    result_img = draw_patches(img.copy(), env.patch_list)
    result_img.save("result.png")
    print(f"Saved result to result.png")

if __name__ == "__main__":
    # 使用你的图片路径和文本提示
    # img_path = "/path/to/your/image.jpg"
    # prompt = "your text prompt"
    # infer(img_path, prompt)
    print("Please uncomment and set img_path and prompt in infer.py to run inference.")