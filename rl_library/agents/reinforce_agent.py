import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import logging
import torch
from gym.spaces import Box
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead

from rl_library.monitors.openai_gym_monitor import GymMonitor

torch.manual_seed(0)  # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99


class ReinforceAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, mode: str = "train", seed: int = 42, hidden_size=10, config: dict = {}):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.mode = mode

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = deque(maxlen=100)  # last 100 scores
        self.avg_loss = np.inf
        self.step_every_action = False

        # For continous problem, for each action we will have as output the mean and the std of the continuous action
        if self.config.get("problem_type") == "continuous":
            action_size = action_size * 2
        self.network = SimpleNeuralNetHead(action_size, SimpleNeuralNetBody(state_size, (hidden_size,)),
                                           func=config.get("head_func"))
        logger.info(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-2)
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
        y = self.network.forward(state).cpu()
        if self.config.get("problem_type") == "discrete":
            m = Categorical(y)
            action = m.sample()
            delta = m.log_prob(action)
            action = action.item()
        elif self.config.get("problem_type") == "continuous":
            action = self.continuous_action(y.detach().numpy()[0])
            delta = torch.tensor([1,])
        self.saved_log_probs.append(delta)
        return action

    @staticmethod
    def continuous_action(y):
        n_actions = int(len(y)/2)
        print(f"n_actions={n_actions}")
        # even indexes contain the mean, odd contain std
        actions = [np.random.normal(y[i*2], y[i*2+1]) for i in range(n_actions)]
        print(f"actions={actions}")
        return actions

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath, mode="train"):
        pass


if __name__ == "__main__":
    # env = GymMonitor("CartPole-v0")  # "CartPole-v0", "MountainCar-v0", "MountainCarContinuous-v0"
    env = GymMonitor("MountainCarContinuous-v0")  # "CartPole-v0", "MountainCar-v0",
    config = {"head_func": F.softmax,
              "problem_type": "continuous" if isinstance(env.action_space, Box) else "discrete"
              }
    agent = ReinforceAgent(state_size=env.state_size, action_size=env.action_size, config=config)

    env.run(agent,
            n_episodes=2000,
            length_episode=500,
            mode="train",
            reload_path=None,
            save_every=500,
            save_path="../../figures")
    env.play(agent)
