# models/rl_agent.py
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from policy_transformer import GridPolicyNet

class SB3Wrapper(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=768):
        super().__init__(obs_space, features_dim)
        self.net = GridPolicyNet(action_dim=9)

    def forward(self, obs):
        logits, value = self.net(obs["img"], obs["prompt_ids"])
        self._value_out = value
        return logits  # SB3 会把它当 feature，接着接动作 head

    def get_value(self):
        return self._value_out
