"""
 Created by quentincangelosi at 17.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
from collections import deque
import gym
import logging
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import os, pickle

from rl_library.agents import DQAgent
from rl_library.agents.base_agent import BaseAgent
from rl_library.utils.visualization import plot_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym-monitor")


class GymMonitor():

    def __init__(self, env_name, threshold=None):
        self.env_name = env_name
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

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.threshold = threshold

    @staticmethod
    def _get_space_size(space) -> int:
        if isinstance(space, Discrete):
            size = space.n
        elif isinstance(space, Box):
            size = space.shape[0]
        else:
            assert f'Unknown space {space}'
        return size

    def run(self, agent, n_episodes=2000, length_episode=500, mode="train", reload_path=None, save_every=500,
            save_path=None):
        ts = pd.Timestamp.utcnow()
        # ------------------------------------------------------------
        #  1. Initialization
        # ------------------------------------------------------------
        # reset the environment
        scores = []  # list containing scores from each episode
        solved = False
        if reload_path is not None and os.path.exists(reload_path + "/checkpoint.pth"):
            logger.info(f"Reloading session from {reload_path}")
            with open(reload_path + "/scores.pickle", "rb") as f:
                scores = pickle.load(f)
        if mode != "train":
            self.eps_start = 0
            self.eps_decay = 0
            self.eps_end = 0

        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0
            actions = []
            rewards = []
            states = [state]
            for t in range(int(length_episode)):
                action = agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)  # send the action to the environment

                if mode == "train" and agent.step_every_action:
                    agent.step(state, action, reward, next_state, done)
                # We are going to update the agent only after the end of the episode
                elif mode == "train":
                    actions.append(action)
                    states.append(next_state)
                    rewards.append(reward)

                state = next_state
                score += reward
                if done:
                    break

            if mode == "train" and not agent.step_every_action:
                agent.step(states, actions, rewards)

            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            self.logging(i_episode, scores_window, score, eps, mode, solved)

            self.intermediate_save(save_every, i_episode, scores, agent, save_path, mode)

        self.save_and_plot(scores, agent, save_path, mode)

        te = pd.Timestamp.utcnow()
        logger.info(f"Elapsed Time: {pd.Timedelta(te - ts)}")
        return scores

    def logging(self, i_episode, scores_window, score, eps, mode, solved):
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}, DQN Avg. Loss: '
              f'{agent.avg_loss:.2e}, Last Score: {score:.2f}, eps: {eps:.2f}', end="")
        if i_episode % 100 == 0:
            logger.info(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}, DQN Avg. Loss: '
                        f'{agent.avg_loss:.2e}, Last Score: {score:.2f}')

        if np.mean(scores_window) >= 13.5 and mode == "train" and not solved:
            logger.warning(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score:'
                           f' {np.mean(scores_window):.2f}')
            solved = True
            # break

    def intermediate_save(self, save_every, i_episode, scores, agent, save_path, mode):
        if save_every and mode == "train" and save_path and i_episode % save_every == 0:
            logger.info(f'\nSaving model to {save_path}')
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            rolling_mean = plot_scores(scores, path=save_path, threshold=self.threshold, prefix=self.env_name)

    def save_and_plot(self, scores, agent, save_path, mode):
        if save_path and mode == "train":
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            rolling_mean = plot_scores(scores, path=save_path, threshold=self.threshold, prefix=self.env_name)
        elif mode == "test":
            rolling_mean = plot_scores(scores, path=".", threshold=self.threshold, prefix=self.env_name)

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

    def play(self, agent):
        state = self.env.reset()
        for t in range(1000):
            action = agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)
            if done:
                break
        self.env.close()


if __name__ == "__main__":
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