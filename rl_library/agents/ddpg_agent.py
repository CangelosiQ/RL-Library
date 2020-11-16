from collections import deque

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
from rl_library.utils.normalizer import MeanStdNormalizer, RunningMeanStd
from rl_library.utils.replay_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("rllib.ddpgagent")


class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config: dict,
                 model_actor: nn.Module, model_critic: nn.Module,
                 actor_target: nn.Module, critic_target: nn.Module,
                 action_space_high, action_space_low, debug_mode=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, config)

        # General class parameters
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.losses = deque(maxlen=int(1e3))  # actor, critic
        self.avg_loss = [0, 0]  # actor, critic
        self.training = True
        self.n_agents = config.get("n_agents", 1)
        self.debug_mode = debug_mode
        self.debug_freq = 4200
        self.debug_it = 1

        # Hyper Parameters
        self.BUFFER_SIZE = config.get("BUFFER_SIZE", int(1e5))  # replay buffer size
        self.BATCH_SIZE = config.get("BATCH_SIZE", 128)  # minibatch size
        self.GAMMA = config.get("GAMMA", 0.99)  # discount factor
        self.TAU = config.get("TAU", 1e-3)  # for soft update of target parameters
        self.LR_ACTOR = config.get("LR_ACTOR", 1e-3)  # learning rate of the actor
        self.LR_CRITIC = config.get("LR_CRITIC", 1e-3)  # learning rate of the critic
        self.WEIGHT_DECAY = config.get("WEIGHT_DECAY", 0.001)  # L2 weight decay
        self.UPDATE_EVERY = config.get("UPDATE_EVERY", 1)  # number of
        # learning steps per environment step
        self.N_CONSECUTIVE_LEARNING_STEPS = config.get("N_CONSECUTIVE_LEARNING_STEPS", 1)  # number of
        # consecutive learning steps during one environment step

        # Actor Network (w/ Target Network)
        self.actor_local = model_actor.to(device)
        self.actor_target = actor_target.to(device)  # copy.deepcopy(model_actor).to(device)

        # Critic Network (w/ Target Network)
        self.critic_local = model_critic.to(device)
        self.critic_target = critic_target.to(device)  #copy.deepcopy(model_critic).to(device)

        # Optimizers
        self.set_optimizer(config.get("optimizer", "adam"))

        # Learning Rate schedulers
        self.set_scheduler(config.get('lr_scheduler'))

        # Normalizers
        self.init_normalizers(config=config)

        # Noises
        self.init_noises(config=config)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # logger.debug(f"State= {state}, Action={action}, Reward={reward}, done={done}")

        # For each agent, save experience / reward
        if len(state.shape) > 1:
            for i in range(state.shape[0]):
                self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        # Start learning only once the number of steps of warm up have been executed
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and self.warmup <= 0 and self.training:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.N_CONSECUTIVE_LEARNING_STEPS):
                    experiences = self.memory.sample()
                    # experiences = self.batch_normalization(experiences)
                    self.learn(experiences, self.GAMMA)

        # Warming-up
        elif self.warmup > 0:
            self.warmup -= 1
            if self.warmup == 0:
                logger.info(f"End of warm up after {len(self.memory)} steps.")

        # Debug mode (activating more logs)
        if self.debug_mode:
            self.debug_it = (self.debug_it + 1) % self.debug_freq

    def batch_normalization(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        if self.state_normalizer:
            states = self.state_normalizer(states)
            next_states = self.state_normalizer(next_states)
            states = torch.from_numpy(states).float().to(device)
            next_states = torch.from_numpy(next_states).float().to(device)

        if self.reward_normalizer:
            rewards = self.reward_normalizer(rewards)
            rewards = torch.from_numpy(rewards).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def warmup_action(self, state):
        # Random action
        if self.n_agents > 1:
            action = [np.random.uniform(self.action_space_low, self.action_space_high) for _ in range(
                self.n_agents)]
        else:
            action = np.random.uniform(self.action_space_low, self.action_space_high)
        return action

    def act(self, state, eps: float = 0):
        """Returns actions for given state as per current policy."""
        # Warm-up = Random action
        if self.warmup > 0:
            if self.n_agents>1:
                action = np.array([eps * self.action_noise[i].sample() for i in range(self.n_agents)])
            else:
                action = self.action_noise[0].sample()
            # action = self.warmup_action(state)
            return action

        if self.state_normalizer:
            if self.state_normalizer.__class__.__name__ == "BatchNorm1d":
                state = self.state_normalizer(state)
            else:
                state = self.state_normalizer(state, read_only=True)

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
            # action = [self.actor_local(state).cpu().data.numpy() for _ in range(self.n_agents)]
            action = self.actor_local(state).cpu().data.numpy()
            if self.debug_mode and self.debug_it % self.debug_freq == 0:
                logger.info(f"DEBUG: act()")
                logger.info(f"State= {state}")
                logger.info(f"Action={action}")

            # if add_noise and self.t_step == 0:
            #     for l, noise in zip(self.actor_local.body.layers, self.parameter_noise):
            #         if l in previous_weights:
            #             l.weight.data = previous_weights[l]

        if self.action_noise is not None and self.training:
            if self.n_agents>1:
                noise = np.array([eps * self.action_noise[i].sample() for i in range(self.n_agents)])
            else:
                noise = self.action_noise[0].sample()

            action += noise
            if self.debug_mode and self.debug_it % self.debug_freq == 0:
                logger.info(f"Noise={noise}")
                logger.info(f"Noisy Action= {action}")

        action = np.clip(action, self.action_space_low, self.action_space_high)
        if self.debug_mode and self.debug_it % self.debug_freq == 0:
            logger.info(f"Clipped Action: {action}")

        # if self.n_agents == 1:
        #     action = action[0]

        self.actor_local.train()
        return action

    def reset(self):
        if self.action_noise:
            for noise in self.action_noise:
                noise.reset()
        if self.parameter_noise:
            self.parameter_noise.reset()

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
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        self.losses.append([float(actor_loss), float(critic_loss)])
        self.avg_loss = np.mean(np.array(self.losses), axis=0)

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

        if self.debug_mode and self.debug_it % self.debug_freq == 0:
            logger.info(f"DEBUG: {self.debug_it}")
            logger.info(f"states={states}")
            logger.info(f"actions={actions}")
            logger.info(f"next_states={next_states}")
            logger.info(f"actions_next={actions_next}")
            logger.info(f"Q_targets_next={Q_targets_next}")
            logger.info(f"Q_expected={Q_expected}")
            logger.info(f"critic_loss={critic_loss}")
            logger.info(f"actions_pred={actions_pred}")
            logger.info(f"actor_loss={actor_loss}")
            logger.info(f"self.critic_local(states, actions_pred)={self.critic_local(states, actions_pred)}")
            logger.info(f"self.critic_local={self.critic_local}")
            logger.info(f"self.critic_target={self.critic_target}")
            logger.info(f"self.actor_local={self.actor_local}")
            logger.info(f"self.actor_target={self.actor_target}")

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

    def set_scheduler(self, config=None):
        """
        Set a scheduler for the learning rate
    
        Parameters
        ----------
        step_size
        gamma
        max_epochs
        scheduler_type
        verbose
        milestones
    
        Returns
        -------
    
        """
        self.actor_scheduler = None
        self.critic_scheduler = None
        if config:

            scheduler_type = config.get("scheduler_type")
            step_size = config.get("step_size", 1)
            gamma = config.get("gamma", 0.9)
            max_epochs = config.get("max_epochs", -1)
            verbose = config.get("verbose", True)
            milestones = config.get("milestones", [50 * i for i in range(1, 6)])

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
                self.actor_scheduler = lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.1,
                                                                      patience=100, threshold=0.0001,
                                                                      threshold_mode='rel',
                                                                      cooldown=0, min_lr=0, eps=1e-08, verbose=verbose)
                self.critic_scheduler = lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.1,
                                                                       patience=100, threshold=0.0001,
                                                                       threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08, verbose=verbose)

            elif scheduler_type == 'multistep':
                self.actor_scheduler = lr_scheduler.MultiStepLR(self.actor_optimizer, milestones, gamma=gamma)
                self.critic_scheduler = lr_scheduler.MultiStepLR(self.critic_optimizer, milestones, gamma=gamma)

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

    def init_normalizers(self, config):
        self.state_normalizer = config.get("state_normalizer")
        self.reward_normalizer = config.get("reward_normalizer")

        if self.state_normalizer == "MeanStd":
            self.state_normalizer = MeanStdNormalizer()
        elif self.state_normalizer == "RunningMeanStd":
            self.state_normalizer = RunningMeanStd(shape=self.state_size)
        elif self.state_normalizer == "BatchNorm":
            self.state_normalizer = nn.BatchNorm1d(self.state_size)

        if self.reward_normalizer == "MeanStd":
            self.reward_normalizer = MeanStdNormalizer()

        logger.info(f"Initiated state_normalizer={self.state_normalizer}, reward_normalizer={self.reward_normalizer}")

    def init_noises(self, config):
        # if self.n_agents >1:
        #     size = (self.n_agents, self.action_size)
        # else:
        size = (self.action_size, )

        # Noise process
        if "action_noise" in config:
            if config["action_noise"] == "OU":
                self.action_noise = [OUNoise(size, self.seed*i, scale=config.get("action_noise_scale",
                                                                                      1)) for i in range(self.n_agents)]
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

    def save(self, filepath):
        checkpoint = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'config': self.config,
            'state_dict_actor_local': self.actor_local.state_dict(),
            'state_dict_actor_target': self.actor_target.state_dict(),
            'state_dict_critic_local': self.critic_local.state_dict(),
            'state_dict_critic_target': self.critic_target.state_dict(),

            'optimizer_actor': self.actor_optimizer.state_dict(),
            'optimizer_critic': self.critic_optimizer.state_dict()
        }

        torch.save(checkpoint, filepath+'/checkpoint.pth')

    def __str__(self):
        s = f"DDPG Agent: \n" \
            f"Actor Optimizer: {self.actor_optimizer}\n" \
            f"Critic Optimizer: {self.critic_optimizer}"
        if self.state_normalizer:
            s += f"\nState Optimizer: {self.state_normalizer}"
        return s
