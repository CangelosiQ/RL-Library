"""
 Created by quentincangelosi at 21.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger("rllib")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class SimpleNeuralNetHead(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, body, func=F.softmax):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(SimpleNeuralNetHead, self).__init__()
        self.body = body
        self.head = nn.Linear(body.layers_sizes[-1], action_size)
        self.func = func
        self.reset_parameters()
        logger.info(f"Initialized {self.__class__.__name__} with body : {self.body.layers} and head {self.head}")

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.body(x)
        x = self.head(x).to(device)
        if self.func:
            if self.func.__name__ in ["softmax",]:
                x = self.func(x, dim=1).to(device)
            else:
                x = self.func(x).to(device)

        return x

    def reset_parameters(self):
        self.body.reset_parameters()


class DeepNeuralNetHeadCritic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, body, func=F.leaky_relu, end_func=F.softmax, hidden_layers_sizes: tuple = (10,)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DeepNeuralNetHeadCritic, self).__init__()
        self.body = body
        self.layers_sizes = (body.layers_sizes[-1]+action_size,) + hidden_layers_sizes + (action_size, )

        self.layers = nn.ModuleList(
            [nn.Linear(inputs, outputs) for inputs, outputs in zip(self.layers_sizes[:-1], self.layers_sizes[1:])])
        logger.info(f"Initialized {self.__class__.__name__} with body : {self.body.layers} and head {self.layers}")
        self.func = func
        self.end_func = end_func

    def forward(self, x, action):
        """Build a network that maps state -> action values."""
        x = self.body(x)
        x = torch.cat((x, action), dim=1)
        for layer in self.layers:
            x = self.func(layer(x)).to(device)

        if self.end_func:
            if self.end_func.__name__ in ["softmax",]:
                x = self.end_func(x, dim=1).to(device)
            else:
                x = self.end_func(x).to(device)

        return x

    def reset_parameters(self):
        self.body.reset_parameters()
        for l in self.layers[:-1]:
            l.weight.data.uniform_(*hidden_init(l))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
