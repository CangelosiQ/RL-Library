import copy
import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logger = logging.getLogger("rllib.agent")
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 5  # how often to update the network
MIN_PROBA_EXPERIENCE = 1e-6  # minimum probability for an experience to be chosen
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyBasedAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, model: nn.Module = None, hidden_layer_sizes: list = None, seed=42,
                 options: list = [], mode: str = "train", post_process_action = None):
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
        self.mode = mode

        # Q-Network
        if model:
            self.qnetwork_local = model.to(device)
            self.qnetwork_target = copy.deepcopy(model).to(device)
        elif hidden_layer_sizes:
            self._init_with_hidden_layer_sizes()
        else:
            architecture = OrderedDict([
                ('fc1', nn.Linear(state_size, action_size)),
                ('softmax', nn.Softmax(dim=1))])
            self.qnetwork_local = nn.Sequential(architecture).to(device)
            self.qnetwork_target = nn.Sequential(architecture).to(device)

        logger.info(f"Initialized model: {self.qnetwork_local}")

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.use_prioritized_replay = "prioritized-replay" in options
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,
                                   use_prioritized_replay=self.use_prioritized_replay)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.losses = deque(maxlen=100)  # last 100 scores
        self.avg_loss = np.inf

        # DQN Improvement Options
        self.use_double_q_learning = "double-q-learning" in options
        if self.use_double_q_learning:
            logger.info("Using Double Q-Learning.")

        self.post_process_action = post_process_action

    def _init_with_hidden_layer_sizes(self):
        layers_sizes = [self.state_size, ] + self.hidden_layers_sizes
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append((f'fc{i}', nn.Linear(layers_sizes[i], layers_sizes[i + 1])))
            layers.append((f'relu{i}', nn.ReLU()), )
        layers.append((f'output', nn.Linear(layers_sizes[-1], self.action_size)))
        layers.append(('softmax', nn.Softmax(dim=1)), )

        architecture = OrderedDict(layers)
        self.qnetwork_local = nn.Sequential(architecture).to(device)
        self.qnetwork_target = nn.Sequential(architecture).to(device)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        if self.use_prioritized_replay:
            proba = self.get_step_proba(state, action, reward, next_state, done)
        else:
            proba = 1

        self.memory.add(state, action, reward, next_state, done, proba)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

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
            actions = np.argmax(action_values.cpu().data.numpy())
        else:
            actions = random.choice(np.arange(self.action_size))

        if self.post_process_action is not None:
            actions = self.post_process_action(actions)
        return actions

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            GAMMA (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, ids = experiences

        # put in train mode
        self.qnetwork_local.train()

        if not self.use_double_q_learning:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                best_actions_local = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            self.qnetwork_local.train()

            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions_local)

        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets).to(device)

        if self.use_prioritized_replay:
            # ------------------- update priorities ------------------- #
            errors = Q_targets-Q_expected
            self.memory.update_priorities(ids, errors=np.hstack(errors.detach().numpy()))
            # ------------------- adjust loss to weights ------------------- #
            loss = (torch.FloatTensor(weights) * loss).mean()

        # logger.info('\rLoss {:.2e}'.format(loss), end="")  # ,
        self.losses.append(float(loss.mean()))
        self.avg_loss = np.mean(self.losses)
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
            target_param.data.copy_((1-tau) * target_param.data + tau * local_param.data)

    def get_step_proba(self, state, action, reward, next_state, done):

        # put in test mode
        self.qnetwork_local.eval()

        state = torch.from_numpy(np.vstack([state])).float().to(device)
        action = torch.from_numpy(np.vstack([action])).long().to(device)
        reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_state = torch.from_numpy(np.vstack([next_state])).float().to(
            device)
        done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(
            device)

        if not self.use_double_q_learning:
            Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                best_actions_local = self.qnetwork_local(next_state).detach().argmax(1).unsqueeze(1)
            self.qnetwork_local.train()

            Q_target_next = self.qnetwork_target(next_state).detach().gather(1, best_actions_local)

        Q_target = reward + (GAMMA * Q_target_next * (1 - done))

        Q_expected = self.qnetwork_local(state).gather(1, action)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_target).to(device)
        loss = loss.detach()
        proba = abs(loss) + MIN_PROBA_EXPERIENCE

        # put in train mode
        self.qnetwork_local.train()

        return proba

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

    @classmethod
    def load(cls, filepath, mode="train"):
        logger.info(f"Loading Agent from {filepath}")
        checkpoint = torch.load(filepath+'/checkpoint.pth')
        agent = cls(state_size=checkpoint["state_size"],
                    action_size=checkpoint["action_size"],
                    hidden_layer_sizes=checkpoint["hidden_layers_sizes"])

        agent.qnetwork_local.load_state_dict(checkpoint['state_dict_local'])
        if mode == "train":
            agent.qnetwork_local.train()
        else:
            agent.qnetwork_local.eval()
        agent.qnetwork_target.load_state_dict(checkpoint['state_dict_target'])
        agent.qnetwork_target.eval()

        agent.optimizer = optim.Adam(agent.qnetwork_local.parameters(), lr=LR)
        agent.optimizer.load_state_dict(checkpoint["optimizer"])

        #TODO save and reload memory?
        return agent



