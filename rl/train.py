import yaml
from envs.env import ClipGridEnv

from stable_baselines3 import PPO

cfg = yaml.safe_load(open("config.yaml"))

env = ClipGridEnv(cfg)

model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    gamma=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    verbose=1,
    device="cuda"
)

model.learn(total_timesteps=cfg["train_steps"])
model.save(cfg["model_save_path"])
