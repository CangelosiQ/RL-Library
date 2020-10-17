"""
 Created by quentincangelosi at 17.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""

import gym
import logging

from gym.spaces import Discrete, Box
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym-monitor")


class GymMonitor():

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        logger.info(f'Actions Space: {self.action_space}')
        logger.info(f'Observation Space: {self.observation_space}')

        # number of actions
        self.action_size = self._get_space_size(self.action_space)
        logger.info(f'Number of actions: {self.action_size}')

        # examine the state space
        self.state_size = self._get_space_size(self.observation_space)
        logger.info(f'States have length: {self.state_size}')

    @staticmethod
    def _get_space_size(space) -> int:
        if isinstance(space, Discrete):
            size = space.n
        elif isinstance(space, Box):
            size = space.shape[0]
        else:
            assert f'Unknown space {space}'
        return size

    def run(self):
        self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == "__main__":

    GymMonitor("CartPole-v0").run()

