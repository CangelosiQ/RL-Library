import numpy as np
import random
from collections import namedtuple, deque

from rl_library.agents.models.model import QNetwork

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
print(f"/////////////////// torch.cuda.is_available() = {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_layer_sizes=[256, 64], seed=42):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers_sizes = hidden_layer_sizes
        self.seed = random.seed(seed)

        # Q-Network
        from collections import OrderedDict
        architecture = OrderedDict([
            ('fc1',  nn.Linear(state_size, 30)),
            ('relu',  nn.ReLU()),
            ('fc2',  nn.Linear(30, 30)),
            ('relu',  nn.ReLU()),
            ('output', nn.Linear(30, action_size)),
            ('softmax', nn.Softmax(dim=1))])
        self.qnetwork_local = nn.Sequential(architecture).to(device)

        # self.qnetwork_local = QNetwork(state_size, action_size,
        #                                hidden_layers_sizes=hidden_layer_sizes,
        #                                seed=seed).to(device)
        print(f"Initialized model: {self.qnetwork_local}")
        self.qnetwork_target = nn.Sequential(architecture).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    @staticmethod
    def preprocess_state(state):
        #return state.flatten()
        return state

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self.preprocess_state(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # put in train mode
        self.qnetwork_local.train()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets).to(device)
        print('\rLoss {:.2e}'.format(loss), end="")  # ,
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath):
        checkpoint = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_layers_sizes': self.hidden_layers_sizes,
            'state_dict_local': self.qnetwork_local.state_dict(),
            'state_dict_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, filepath+'/checkpoint.pth')

    def load(self, filepath):

        checkpoint = torch.load(filepath+'/checkpoint.pth')
        self.qnetwork_local = QNetwork(checkpoint["state_size"],
                                       checkpoint["action_size"],
                                       checkpoint["hidden_layers_sizes"]).to(device)

        self.qnetwork_local.load_state_dict(checkpoint['state_dict_local'])
        self.qnetwork_local.train()

        self.qnetwork_target = QNetwork(checkpoint["state_size"],
                                        checkpoint["action_size"],
                                        checkpoint["hidden_layers_sizes"]).to(device)
        self.qnetwork_target.load_state_dict(checkpoint['state_dict_target'])
        self.qnetwork_target.eval()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
