import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class SimpleNeuralNetBodyCritic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, hidden_layers_sizes: tuple = (10, ), func= F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(SimpleNeuralNetBodyCritic, self).__init__()
        self.layers_sizes = (state_size,) + hidden_layers_sizes

        self.layers = nn.ModuleList(
            [nn.Linear(inputs+4*(i==1), outputs) for i, (inputs, outputs) in enumerate(zip(self.layers_sizes[:-1],
                                                                               self.layers_sizes[
                                                                                                 1:]))])

        self.func = func

    def forward(self, x, action):
        """Build a network that maps state -> action values."""
        xs = self.func(self.layers[0](x))
        x = torch.cat((xs, action), dim=1)
        for layer in self.layers[1:]:
            x = self.func(layer(x)).to(device)
        return x