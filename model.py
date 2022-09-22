import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# define the actor network
class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(args.seed)
        self.fc1 = nn.Linear(args.obs_shape, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100,args.action_shape)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

