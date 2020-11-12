import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import logging

from rl_library.agents.base_agent import BaseAgent
from rl_library.utils.noises import OUNoise
from rl_library.utils.normalizer import MeanStdNormalizer
from rl_library.utils.replay_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("rllib.ddpgagent")


class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config: dict,
                 model_actor: nn.Module, model_critic: nn.Module,
                 action_space_high, action_space_low):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, config)

        self.BUFFER_SIZE = config.get("BUFFER_SIZE", int(1e5))  # replay buffer size
        self.BATCH_SIZE = config.get("BATCH_SIZE", 128)  # minibatch size
        self.GAMMA = config.get("GAMMA", 0.99)  # discount factor
        self.TAU = config.get("TAU", 1e-3)  # for soft update of target parameters
        self.LR_ACTOR = config.get("LR_ACTOR", 1e-3)  # learning rate of the actor
        self.LR_CRITIC = config.get("LR_CRITIC", 1e-3)  # learning rate of the critic
        self.WEIGHT_DECAY = config.get("WEIGHT_DECAY", 0.001)  # L2 weight decay
        self.UPDATE_EVERY = config.get("UPDATE_EVERY", 1)

        # Actor Network (w/ Target Network)
        self.actor_local = model_actor.to(device)
        self.actor_target = copy.deepcopy(model_actor).to(device)

        # Critic Network (w/ Target Network)
        self.critic_local = model_critic.to(device)
        self.critic_target = copy.deepcopy(model_critic).to(device)

        self.set_optimizer(config.get("optimizer", "adam"))

        if config.get('lr_scheduler'):
            self.set_scheduler(**config.get('lr_scheduler'))

        # Noise process
        if "action_noise" in config:
            if config["action_noise"] == "OU":
                self.action_noise = OUNoise(self.action_size, self.random_seed, scale=config.get(
                    "action_noise_scale", 1))
            else:
                logger.warning(f"action_noise {config['action_noise']} not understood.")
                self.action_noise = None
        else:
            self.action_noise = None

        if "parameter_noise" in config:
            if config["parameter_noise"] == "OU":
                self.parameter_noise = [OUNoise(l.weight.data.size(), self.random_seed) for l in
                                        self.actor_local.body.layers]
            else:
                logger.warning(f"action_noise {config['action_noise']} not understood.")
                self.parameter_noise = None
        else:
            self.parameter_noise = None

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.random_seed)
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high

        self.avg_loss = [0, 0]  # actor, critic

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
                # experiences = self.batch_normalization(experiences)
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

    def act(self, state, eps: float = 0):
        """Returns actions for given state as per current policy."""
        if self.warmup > 0:
            # Random action
            action = np.random.uniform(self.action_space_low, self.action_space_high)
            return action
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            # if add_noise and self.t_step == 0:
            #     previous_weights = {}
            #     for l, noise in zip(self.actor_local.body.layers, self.parameter_noise):
            #         #         # print(f"before {np.mean(l.weight.data.numpy())}")
            #         if random.random() < eps:  # Only affect some layers at a time
            #             # l.weight.data += eps/10 * (torch.tensor(np.random.random(l.weight.data.size())-0.5))
            #             # std = max(min(np.std(l.weight.data.numpy()), 1), 0.1)
            #             # random_noise = eps * (torch.tensor((np.random.random(l.weight.data.numpy().shape)-0.5)*std))
            #             random_noise = eps * (torch.tensor(noise.sample()))
            #             # print(f"l.weight.data: {l.weight.data}")
            #             # print(f"Noise: {random_noise}")
            #             previous_weights[l] = copy.deepcopy(l.weight.data)
            #             l.weight.data += random_noise

            # print(f"after {np.mean(l.weight.data.numpy())}")
            action = self.actor_local(state).cpu().data.numpy()
            # if add_noise and self.t_step == 0:
            #     for l, noise in zip(self.actor_local.body.layers, self.parameter_noise):
            #         if l in previous_weights:
            #             l.weight.data = previous_weights[l]

        # logger.debug(f"State= {state}, Action={action}")

        if self.action_noise is not None:
            noise = self.action_noise.sample()
            # noise = 2*np.random.random(action.shape) - 1
            action += eps * noise
            # logger.debug(f"Noisy Action={action} Noise={noise}")
        action = np.clip(action, self.action_space_low, self.action_space_high)
        # logger.debug(f"Clipped Action: {action}")

        self.actor_local.train()
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
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 5)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.avg_loss = [float(actor_loss), float(critic_loss)]

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        # ----------------------- update learning rates ----------------------- #
        # TODO: this could also be done once per episode. It's normally done once per epoch in DL. One argument
        #  against having it per episode is for continuous tasks with no episode concept. although it seems that
        #  there is always an "episode concept"
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def enable_evaluation_mode(self):
        self.action_noise_backup = self.action_noise
        self.param_noise_backup = self.parameter_noise
        self.action_noise = None
        self.param_noise = None

    def disable_evaluation_mode(self):
        self.action_noise = self.action_noise_backup
        self.param_noise = self.param_noise_backup

    def set_scheduler(self, step_size: int = 1, gamma: float = 0.9,
                      max_epochs: int = -1, scheduler_type: str = 'step', verbose=True):
        """
        Set a scheduler for the learning rate
    
        Parameters
        ----------
        step_size
        gamma
        max_epochs
        scheduler_type
    
        Returns
        -------
    
        """

        def _decay(max_epochs, gamma):
            lambda_lr = lambda iter: (1 - iter / max_epochs) ** gamma
            return lambda_lr

        if scheduler_type == 'step':
            self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=step_size, gamma=gamma, )
            self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=step_size, gamma=gamma,
                                                        )
        elif scheduler_type == 'exp':
            self.actor_scheduler = lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=gamma, )
            self.critic_scheduler = lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=gamma, )
        elif scheduler_type == 'decay':
            self.actor_scheduler = lr_scheduler.LambdaLR(self.actor_optimizer, [_decay(max_epochs, gamma)], )
            self.critic_scheduler = lr_scheduler.LambdaLR(self.critic_optimizer, [_decay(max_epochs, gamma)], )
        elif scheduler_type == 'plateau':
            self.actor_scheduler = lr_scheduler.LambdaLR(self.actor_optimizer, [_decay(max_epochs, gamma)], )
            self.critic_scheduler = lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.1,
                                                                   patience=100, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        logger.info(f"Actor LR Scheduler: {self.actor_scheduler}")
        logger.info(f"Critic LR Scheduler: {self.critic_scheduler}")

    def set_optimizer(self, name: str):
        """
        Set the optimizer

        Parameters
        ----------
        name
        learning_rate

        Returns
        -------

        """
        if name == 'sgd':
            self.actor_optimizer = optim.SGD(self.actor_local.parameters(), lr=self.LR_ACTOR, momentum=0.9,
                                             weight_decay=self.WEIGHT_DECAY)
            self.critic_optimizer = optim.SGD(self.critic_local.parameters(), lr=self.LR_CRITIC, momentum=0.9,
                                              weight_decay=self.WEIGHT_DECAY)
        elif name == 'adam':
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR,
                                              weight_decay=self.WEIGHT_DECAY)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                               weight_decay=self.WEIGHT_DECAY)
        else:
            raise ValueError(f'Optimizer not supported: {name}. Current options: [sgd, adam]')

        logger.info(f"Actor Optimizer: {self.actor_optimizer}")
        logger.info(f"Critic Optimizer: {self.critic_optimizer}")

    def __str__(self):
        s = f"DDPG Agent: \n" \
            f"Actor Optimizer: {self.actor_optimizer}\n" \
            f"Critic Optimizer: {self.critic_optimizer}"
        return s
