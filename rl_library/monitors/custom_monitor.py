"""
 Created by quentincangelosi at 17.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import logging
from gym.spaces import Discrete, Box
import numpy as np
from rl_library.monitors.base_monitor import Monitor

logger = logging.getLogger("rllib.custom-monitor")


class CustomMonitor(Monitor):

    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        logger.info(f'Actions Space: {self.action_space}')
        # logger.info(f'Observation Space: {self.observation_space}')

        # number of actions
        self.action_size = self.env.action_size
        logger.info(f'Number of actions: {self.action_size}')

        # examine the state space
        self.state_size = self.env.state_size
        logger.info(f'States have length: {self.state_size}')

    def reset_env(self):
        return self.env.reset()

    def env_step(self, action):
        logger.debug(f"\rAction: {action} Processed: {self.process_action(action)}")
        next_state, reward, done, _ = self.env.step(self.process_action(action))
        logger.debug(f"Next state = {next_state}")
        logger.debug(f"Reward = {reward}")
        logger.debug(f"Done = {done}")
        return next_state, reward, done, _

    def play(self, agent):
        state = self.env.reset()
        for _ in range(2000):
            action = agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)
            if done:
                break
        self.env.close()

    def process_action(self, actions):
        if isinstance(self.action_space, Box) and "__len__" not in actions.__dir__():
            actions = [actions,]
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions

