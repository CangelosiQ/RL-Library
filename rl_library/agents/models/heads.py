"""
 Created by quentincangelosi at 21.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger("rllib.models")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


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
        self.seed = np.random.seed(seed)
        self.body = body
        self.head = nn.Linear(body.layers_sizes[-1], action_size)
        self.func = func
        self.bn = None # For batch normalization
        self.reset_parameters()
        logger.info(f"Initialized {self.__class__.__name__} with body : {self.body.layers} and head {self.head}")
        logger.info(f"state_dict= {self.state_dict()}")

    def forward(self, x):
        """Build a network that maps state -> action values."""
        if self.bn:
            x = self.bn(x)
        x = self.body(x)
        x = self.head(x).to(device)

        if self.func:
            if self.func.__name__ in ["softmax", ]:
                x = self.func(x, dim=1).to(device)
            else:
                x = self.func(x).to(device)

        return x

    def reset_parameters(self):
        for l in self.body.layers:
            l.weight.data.uniform_(*hidden_init(l))

        self.head.weight.data.uniform_(-3e-3, 3e-3)


class DeepNeuralNetHeadCritic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, body, func=F.leaky_relu, end_func=F.softmax, hidden_layers_sizes: tuple = (10,),
                 seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DeepNeuralNetHeadCritic, self).__init__()
        self.seed = np.random.seed(seed)
        self.body = body
        self.layers_sizes = (body.layers_sizes[-1] + action_size,) + tuple(hidden_layers_sizes) + (1,) #(action_size,)

        self.layers = nn.ModuleList(
            [nn.Linear(inputs, outputs) for inputs, outputs in zip(self.layers_sizes[:-1], self.layers_sizes[1:])])
        self.bn = None  # For batch normalization
        self.reset_parameters()
        logger.info(f"Initialized {self.__class__.__name__} with body : {self.body.layers} and head {self.layers}")
        logger.info(f"state_dict= {self.state_dict()}")
        self.func = func
        self.end_func = end_func

    def forward(self, x, action):
        """Build a network that maps state -> action values."""
        if self.bn:
            x = self.bn(x)
        x = self.body(x)
        # print(f"self.body(x)={x}")
        x = torch.cat((x, action), dim=1)
        # print(f"torch.cat((x, action), dim=1)={x}")
        for layer in self.layers[:-1]:
            x = self.func(layer(x)).to(device)
            # print(f"{layer}: self.func(layer(x))={x}")

        x = self.layers[-1](x).to(device)
        # print(f"{self.layers[:-1]}(x)={x}")
        if self.end_func:
            if self.end_func.__name__ in ["softmax", ]:
                x = self.end_func(x, dim=1).to(device)
            else:
                x = self.end_func(x).to(device)
        # print(f"final x={x}")
        return x

    def reset_parameters(self):

        # Init Body layers
        for l in self.body.layers:
            l.weight.data.uniform_(*hidden_init(l))

        # Init Head Layers
        for l in self.layers[:-1]:
            l.weight.data.uniform_(*hidden_init(l))

        # Init last layers
        # The final layer weights and biases of both the actor and critic
        # were initialized from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4] for the
        # low dimensional and pixel cases respectively. This was to ensure the initial outputs for the policy
        # and value estimates were near zero.
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
