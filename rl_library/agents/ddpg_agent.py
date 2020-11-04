import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import logging

from rl_library.agents.base_agent import BaseAgent
from rl_library.utils.noises import OUNoise
from rl_library.utils.normalizer import MeanStdNormalizer
from rl_library.utils.replay_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("rllib.ddpgagent")


class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, model_actor: nn.Module, model_critic: nn.Module, action_space_high, action_space_low,
                 hyper_parameters: dict, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(**kwargs)

        self.BUFFER_SIZE = hyper_parameters.get("BUFFER_SIZE", int(1e5))  # replay buffer size
        self.BATCH_SIZE = hyper_parameters.get("BATCH_SIZE", 128)  # minibatch size
        self.GAMMA = hyper_parameters.get("GAMMA", 0.99)  # discount factor
        self.TAU = hyper_parameters.get("TAU", 1e-3)  # for soft update of target parameters
        self.LR_ACTOR = hyper_parameters.get("LR_ACTOR", 1e-3)  # learning rate of the actor
        self.LR_CRITIC = hyper_parameters.get("LR_CRITIC", 1e-3)  # learning rate of the critic
        self.WEIGHT_DECAY = hyper_parameters.get("WEIGHT_DECAY", 0.001)  # L2 weight decay
        self.UPDATE_EVERY = hyper_parameters.get("UPDATE_EVERY", 1)

        # Actor Network (w/ Target Network)
        self.actor_local = model_actor.to(device)
        self.actor_target = copy.deepcopy(model_actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = model_critic.to(device)
        self.critic_target = copy.deepcopy(model_critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.action_size, self.random_seed)
        self.parameter_noise = [OUNoise(l.weight.data.size(), self.random_seed) for l in self.actor_local.body.layers]
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.random_seed)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # logger.debug(f"State= {state}, Action={action}, Reward={reward}, done={done}")
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and self.warmup <= 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                experiences = self.batch_normalization(experiences)
                self.learn(experiences, self.GAMMA)
        elif self.warmup > 0:
            self.warmup -= 1
            if self.warmup == 0:
                logger.info(f"End of warm up after {len(self.memory)} steps.")

    @staticmethod
    def batch_normalization(experiences):
        states, actions, rewards, next_states, dones = experiences
        normalizer = MeanStdNormalizer()
        states = normalizer(states)
        next_states = normalizer(next_states, read_only=True)
        rewards = normalizer(rewards)
        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def act(self, state, add_noise=True, eps: float = 1, **kwargs):
        """Returns actions for given state as per current policy."""
        if self.warmup > 0:
            # Random action
            action = np.random.uniform(self.action_space_low, self.action_space_high)
            return action
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            if add_noise and self.t_step == 0:
                previous_weights = {}
                for l, noise in zip(self.actor_local.body.layers, self.parameter_noise):
                    #         # print(f"before {np.mean(l.weight.data.numpy())}")
                    if random.random() < eps:  # Only affect some layers at a time
                        # l.weight.data += eps/10 * (torch.tensor(np.random.random(l.weight.data.size())-0.5))
                        # std = max(min(np.std(l.weight.data.numpy()), 1), 0.1)
                        # random_noise = eps * (torch.tensor((np.random.random(l.weight.data.numpy().shape)-0.5)*std))
                        random_noise = eps * (torch.tensor(noise.sample()))
                        # print(f"l.weight.data: {l.weight.data}")
                        # print(f"Noise: {random_noise}")
                        previous_weights[l] = copy.deepcopy(l.weight.data)
                        l.weight.data += random_noise

                    # print(f"after {np.mean(l.weight.data.numpy())}")
            action = self.actor_local(state).cpu().data.numpy()
            if add_noise and self.t_step == 0:
                for l, noise in zip(self.actor_local.body.layers, self.parameter_noise):
                    if l in previous_weights:
                        l.weight.data = previous_weights[l]

        # logger.debug(f"State= {state}, Action={action}")
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            # noise = np.random.random(action.shape)
            # action += eps*(2*noise-1)
            action += eps * noise
            # logger.debug(f"Noisy Action={action} Noise={noise}")
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # logger.debug(f"Clipped Action: {action}")
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.avg_loss = actor_loss + critic_loss
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
