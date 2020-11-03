"""
 Created by quentincangelosi at 17.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import gym
import logging
from gym.spaces import Discrete, Box
import numpy as np
from rl_library.monitors.base_monitor import Monitor

logger = logging.getLogger("rllib.gym-monitor")


class GymMonitor(Monitor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = gym.make(self.env_name)
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

    def reset(self):
        return self.env.reset()

    def env_step(self, action):
        next_state, reward, done, _ = self.env.step(self.process_action(action))
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



if __name__ == "__main__":
    from rl_library.agents import DQAgent
    from rl_library.agents.base_agent import BaseAgent
    env = GymMonitor("CartPole-v0")
    agent = BaseAgent(state_size=env.state_size, action_size=env.action_size)
    agent = DQAgent(state_size=env.state_size, action_size=env.action_size)

    env.run(agent,
            n_episodes=2000,
            length_episode=500,
            mode="train",
            reload_path=None,
            save_every=500,
            save_path="../../figures")
    env.play(agent)