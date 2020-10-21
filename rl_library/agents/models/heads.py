"""
 Created by quentincangelosi at 21.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleNeuralNetHead(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, body, func=F.softmax, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(SimpleNeuralNetHead, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.body = body
        self.head = nn.Linear(body.layers_sizes[-1], action_size)
        self.func = func

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.body(x)
        x = self.head(x).to(device)
        if self.func:
            x = self.func(x, dim=1).to(device)
        return x
