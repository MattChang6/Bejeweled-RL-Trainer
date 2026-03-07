from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    target_update: int = 500
    replay_size: int = 10000
    min_replay: int = 500


class DQN(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
