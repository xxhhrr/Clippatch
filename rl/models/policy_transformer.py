# models/policy_transformer.py
import torch, torch.nn as nn

class GridPolicyNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.policy = nn.Sequential(
            nn.Linear(32 * 26 * 26, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.value  = nn.Sequential(
            nn.Linear(32 * 26 * 26, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        features = self.cnn(obs)
        return self.policy(features), self.value(features)
