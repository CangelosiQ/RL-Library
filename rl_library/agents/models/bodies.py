import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class SimpleNeuralNetBody(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, hidden_layers_sizes: tuple = (10, ), func= F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(SimpleNeuralNetBody, self).__init__()
        self.layers_sizes = (state_size,) + hidden_layers_sizes

        self.layers = nn.ModuleList(
            [nn.Linear(inputs, outputs) for inputs, outputs in zip(self.layers_sizes[:-1], self.layers_sizes[1:])])

        self.func = func

    def forward(self, x):
        """Build a network that maps state -> action values."""
        for layer in self.layers:
            x = self.func(layer(x)).to(device)
        return x



