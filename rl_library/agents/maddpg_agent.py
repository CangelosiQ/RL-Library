import os
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
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.utils.noises import OUNoise, NormalActionNoise
from rl_library.utils.normalizer import MeanStdNormalizer, RunningMeanStd
from rl_library.utils.replay_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("rllib.ddpgagent")


class MADDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config: dict,
                 model_actor: nn.Module, model_critic: nn.Module,
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
        self.config=config
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.losses = deque(maxlen=int(1e3))  # actor, critic
        self.avg_loss = [0, 0]  # actor, critic
        self.std_loss = [0, 0]  # actor, critic
        self.training = True
        self.n_agents = config.get("n_agents", 1)
        self.debug_mode = debug_mode
        self.debug_freq = 100000
        self.debug_it = 1
        self.BUFFER_SIZE = config.get("BUFFER_SIZE", int(1e5))  # replay buffer size
        self.BATCH_SIZE = config.get("BATCH_SIZE", 128)  # minibatch size
        self.GAMMA = config.get("GAMMA", 0.99)  # discount factor
        self.UPDATE_EVERY = config.get("UPDATE_EVERY", 1)  # number of
        # learning steps per environment step
        self.N_CONSECUTIVE_LEARNING_STEPS = config.get("N_CONSECUTIVE_LEARNING_STEPS", 1)  # number of
        # consecutive learning steps during one environment step

        config["n_agents"] = 1
        self.agents = [DDPGAgent(state_size, action_size, config, model_actor, model_critic, action_space_high,
                                 action_space_low, debug_mode) for _ in range(self.n_agents)]
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self.random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # logger.debug(f"State= {state}, Action={action}, Reward={reward}, done={done}")

        # For each agent, save experience / reward
        # print(f"memory.add(state={state.shape},action={action.shape},reward={reward},done={done},)")
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        # Start learning only once the number of steps of warm up have been executed
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and self.warmup <= 0 and self.training:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.N_CONSECUTIVE_LEARNING_STEPS):
                    experiences = self.memory.sample()
                    # experiences = self.batch_normalization(experiences) TODO
                    self.learn(experiences)

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
            # states = torch.from_numpy(states).float().to(device)
            # next_states = torch.from_numpy(next_states).float().to(device)

        if self.reward_normalizer:
            rewards = self.reward_normalizer(rewards)
            rewards = torch.from_numpy(rewards).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def act(self, state, eps: float = 0):
        """Returns actions for given state as per current policy."""
        # print(f"state={state.shape}")
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(state[i, :], eps)
            # print(action)
            actions.append(action)
        actions = np.array(actions)
        # print("actions:", actions)
        # print(actions.shape)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences):
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
        # print(f"SAMPLE:"
        #       f"\nstates={states.size()}"
        #       f"\nactions={actions.size()}"
        #       f"\nnext_states={next_states.size()}"
        #       f"\ndones={dones.size()}"
        #       f"\nrewards={rewards.size()}"
        #       )
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models FOR EACH AGENT
        actions_next = []
        actions_pred = []
        # with torch.no_grad():
        for j, agent in enumerate(self.agents):
            agent_actions_next = agent.actor_target(next_states[:, j, :])
            agent_actions_pred = agent.actor_local(states[:, j, :])
            actions_next.append(agent_actions_next.unsqueeze(1))
            actions_pred.append(agent_actions_pred.unsqueeze(1))

        all_actions_next = torch.cat(actions_next, dim=1)
        all_actions_next = torch.reshape(all_actions_next, (all_actions_next.size(0), -1))

        for i, agent in enumerate(self.agents):

            all_actions = torch.reshape(actions, (actions.size(0), -1))
            all_states = torch.reshape(states, (states.size(0), -1))
            all_next_states = torch.reshape(next_states, (next_states.size(0), -1))
            # print(f"all_actions= {all_actions.size()}")
            # print(f"all_actions_next= {all_actions_next.size()}")
            # print(f"all_states.size()={all_states.size()}")
            # print(f"all_next_states.size()={all_next_states.size()}")

            # print(f"states[:, i, :]={states[:, i, :].size()}")
            # print(f"actions[:, i, :]={actions[:, i, :].size()}")
            # print(f"rewards[:, i]={rewards[:, i].size()}")
            # print(f"dones[:, i]={dones[:, i].size()}")
            with torch.no_grad():
                Q_targets_next = agent.critic_target(all_next_states, all_actions_next)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, i] + (agent.GAMMA * Q_targets_next * (1 - dones[:, i]))

            # Compute critic loss
            Q_expected = agent.critic_local(all_states, all_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # detach actions from other agents
            actions_pred = [actions if j == i else actions.detach() for j, actions in enumerate(actions_pred)]
            all_actions_pred = torch.cat(actions_pred, dim=1).to(device)
            all_actions_pred = torch.reshape(all_actions_pred, (all_actions_pred.size(0), -1))

            # Compute actor loss
            actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
            # Minimize the loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 1)
            agent.actor_optimizer.step()
            agent.losses.append([float(actor_loss), float(critic_loss)])
            agent.avg_loss = np.mean(np.array(agent.losses), axis=0)

            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target)
            agent.soft_update(agent.actor_local, agent.actor_target)

            # ----------------------- update learning rates ----------------------- #
            # TODO: this could also be done once per episode. It's normally done once per epoch in DL. One argument
            #  against having it per episode is for continuous tasks with no episode concept. although it seems that
            #  there is always an "episode concept"
            if agent.actor_scheduler is not None:
                agent.actor_scheduler.step()
            if agent.critic_scheduler is not None:
                agent.critic_scheduler.step()

            if self.debug_mode and self.debug_it % self.debug_freq == 0:
                logger.info(f"DEBUG: AGENT {i}")
                logger.info(f"states={states}")
                logger.info(f"actions={actions}")
                logger.info(f"next_states={next_states}")
                logger.info(f"actions_next={actions_next}")
                logger.info(f"Q_targets_next={Q_targets_next}")
                logger.info(f"Q_expected={Q_expected}")
                logger.info(f"critic_loss={critic_loss}")
                logger.info(f"actions_pred={actions_pred}")
                logger.info(f"actor_loss={actor_loss}")
                logger.info(f"agent.critic_local(states, actions_pred)={agent.critic_local(states, actions_pred)}")
                logger.info(f"agent.critic_local={agent.critic_local}")
                logger.info(f"agent.critic_target={agent.critic_target}")
                logger.info(f"agent.actor_local={agent.actor_local}")
                logger.info(f"agent.actor_target={agent.actor_target}")
        self.avg_loss = np.mean([agent.avg_loss for agent in self.agents], axis=0)

    def save(self, filepath):
        for i, agent in enumerate(self.agents):
            _path = f'{filepath}/agent_{i}'
            os.makedirs(_path, exist_ok=True)
            agent.save(_path)

    def load(self, filepath, mode="test"):
        for i, agent in enumerate(self.agents):
            agent.load(f'{filepath}/agent_{i}')

    def __str__(self):
        s = f"MADDPG Agent: \n"
        for i, agent in enumerate(self.agents):
            s += f"Agent {i}:" \
                 f"\t- Actor Optimizer: {agent.actor_optimizer}\n" \
                 f"\t- Critic Optimizer: {agent.critic_optimizer}"
            if agent.state_normalizer:
                s += f"\t- State Optimizer: {agent.state_normalizer}"
        return s
