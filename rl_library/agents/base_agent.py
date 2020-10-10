import copy
import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
from rl_library.agents.models.model import QNetwork

logger = logging.getLogger()
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 5  # how often to update the network
MIN_PROBA_EXPERIENCE = 1e-6  # minimum probability for an experience to be chosen
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, mode: str = "train", seed: int = 42):
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

    def step(self, state, action, reward, next_state, done):
        pass

    @staticmethod
    def preprocess_state(state):
        #return state.flatten()
        return state

    def act(self, state, eps=0.):
        pass

    def learn(self, experiences):
        pass

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath, mode="train"):
        pass
