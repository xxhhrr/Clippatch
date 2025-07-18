import yaml, torch
from envs.env import ClipGridEnv
from models.rl_agent   import SB3Wrapper
from stable_baselines3 import PPO

cfg = yaml.safe_load(open("config.yaml"))

env = ClipGridEnv(
    img_paths=cfg["data"]["train_images"],
    prompts  =cfg["data"]["prompts"],
    gt_masks =cfg["data"]["masks"]
)

model = PPO(
    policy=SB3Wrapper,
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    gamma=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    verbose=1,
    device="cuda"
)
model.learn(total_timesteps=cfg["train_steps"])
model.save("grid_policy.pt")
