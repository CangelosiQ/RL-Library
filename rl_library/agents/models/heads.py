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

    def __init__(self, state_size, action_size, hidden_layers_sizes:list = None, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(SimpleNeuralNetBody, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers_sizes = [state_size,] + hidden_layers_sizes if hidden_layers_sizes else [state_size,]

        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        if hidden_layers_sizes is not None:
            self.first_layer = nn.Linear(state_size, hidden_layers_sizes[0]).to(device)
            n_hidden_layers = len(hidden_layers_sizes)
            if n_hidden_layers > 1:
                self.hidden_layers = [nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]).to(device) for i in range(n_hidden_layers-1)]
            self.last_layer = nn.Linear(hidden_layers_sizes[-1], action_size).to(device)
        else:
            self.first_layer = nn.Linear(state_size, action_size).to(device)
            self.hidden_layers = []
            self.last_layer = None

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.first_layer(state)).to(device)
        for layer in self.hidden_layers:
            x = F.relu(layer(x)).to(device)
        if self.last_layer:
            x = self.last_layer(x)
        return x
