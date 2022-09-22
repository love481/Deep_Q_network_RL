import gym
import numpy as np
import torch 
from enum import Enum
import random
import copy
class GaussianNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, sigma=0.2, decay=0.995):
        """Initialize parameters and noise process."""
        self.size = size
        self.sigma = sigma
        self.decay = decay
        self.seed = torch.manual_seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.sigma = max(self.decay*self.sigma,0.01)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return self.sigma * torch.randn(self.size)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2, sigma_min = 0.01, sigma_decay=0.995):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


def make_env(args):
    env = gym.make('LunarLander-v2')
    env.seed(2)
    args.obs_shape = 8
    args.action_shape = 4
    args.high_action = 1
    args.low_action = -1
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.seed=2
    return env, args
