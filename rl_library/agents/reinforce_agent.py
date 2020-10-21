import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import logging
import torch

from rl_library.monitors.openai_gym_monitor import GymMonitor

torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99

class ReinforceAgent(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, mode: str = "train", seed: int = 42, hidden_size=10):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.mode = mode

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = deque(maxlen=100)  # last 100 scores
        self.avg_loss = np.inf

        self.network =
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.optimizer = optim.Adam(ReinforceAgent.parameters(), lr=1e-2)
        self.saved_log_probs = []

    def step(self, states, actions, rewards):
        discounts = [GAMMA ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # reset saved_log_probs
        self.saved_log_probs = []

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath, mode="train"):
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    env = GymMonitor("CartPole-v0")
    agent = ReinforceAgent(state_size=env.state_size, action_size=env.action_size)

    env.run(agent,
            n_episodes=2000,
            length_episode=500,
            mode="train",
            reload_path=None,
            save_every=500,
            save_path="../../figures")
    env.play(agent)