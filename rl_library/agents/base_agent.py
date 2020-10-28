import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import logging

logger = logging.getLogger('rl-lib')

class BaseAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed: int = 42):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.random_seed = random.seed(random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = deque(maxlen=100)  # last 100 scores
        self.avg_loss = np.inf
        self.step_every_action = True

    def step(self, state, action, reward, next_state, done):
        pass

    @staticmethod
    def preprocess_state(state):
        # return state.flatten()
        return state

    def act(self, state, eps=0.):
        return np.random.choice(range(self.action_size))

    def learn(self, experiences):
        pass

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath, mode="train"):
        pass
