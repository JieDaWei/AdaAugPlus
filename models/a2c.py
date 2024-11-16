import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torchvision.transforms as transforms
from torch.distributions import Categorical

class ActorCriticV5(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticV5, self).__init__()
        # Define the shared feature extractor (CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Define the actor network
        self.actor = nn.Sequential(
            nn.Linear(50176, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, num_actions),
            nn.Softmax()  # Assumes a continuous action space between -1 and 1
        )
        # Define the critic network
        self.critic = nn.Sequential(
            nn.Linear(50176, 256),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        # print(features.shape)
        return self.actor(features), self.critic(features)