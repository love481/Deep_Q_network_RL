import numpy as np
import torch
from torch import  nn
import os
from dqn import DQN
from utils import OUNoise,GaussianNoise
import random
class Agent(nn.Module):
    def __init__(self,args,agent_id=0):
        super().__init__()
        self.args = args
        print(self.args.lr_actor)
        self.policy = DQN(args,agent_id)
        self.seed = random.seed(args.seed)
        self.noise=OUNoise(args.action_shape,args.seed)

    def select_action(self, o,epsilon=0):
        state = torch.from_numpy(o).float().unsqueeze(0).to(self.args.device)
        self.policy.q_network.eval()
        with torch.no_grad():
            action_values =  self.policy.q_network(state).squeeze(0).cpu().data.numpy()
        self.policy.q_network.train()

        # if epsilon>0:
        #     action_values += self.noise.sample()
        # else:
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action_values = np.argmax(action_values)
        else:
            action_values = random.choice(np.arange(self.args.action_shape))
        # print(action_values)
        return action_values

    def learn(self, experiences):
        self.policy.train(experiences)


