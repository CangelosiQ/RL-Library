"""
 Created by quentincangelosi at 03.11.20
 From Global Advanced Analytics and Artificial Intelligence
"""
from collections import deque
import logging
import numpy as np
import pandas as pd
import os, pickle

from rl_library.utils.visualization import plot_scores

logger = logging.getLogger("rllib.gym-monitor")


class Monitor:

    def __init__(self, env_name, threshold=None):
        self.env_name = env_name
        self.env = None
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.9995
        self.threshold = threshold
        self.agent_losses = []

    def reset(self):
        raise NotImplementedError

    def env_step(self, action):
        raise NotImplementedError

    def run(self, agent, n_episodes=2000, length_episode=500, mode="train", reload_path=None, save_every=500,
            save_path=None, render=False):
        save_prefix = f'{self.env_name}_{agent.__class__.__name__}'
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
        t = None
        for i_episode in range(1, n_episodes + 1):
            state = self.reset()
            last_actions = deque(maxlen=100)
            last_states = deque(maxlen=100)
            score = 0
            actions = []
            rewards = []
            states = [state]
            for t in range(int(length_episode)):
                action = agent.act(state, eps=eps, add_noise=True)
                last_actions.append(action)
                last_states.append(state)
                if len(last_actions) == 100:
                    diff = np.sum(np.std(last_actions, 0))
                    diff_state = np.sum(np.std(last_states, 0))
                    if diff < 0.4 and diff_state < 0.2:
                        print(f"Episode {i_episode}: AGENT STUCK! diff={diff}, diff_state={diff_state}")
                        break

                # print(f"\rAction: {action} Processed: {self.process_action(action)}", end="")
                next_state, reward, done, _ = self.env_step(self.process_action(action))  # send the action to the
                # environment
                if render:
                    self.env.render()

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
            solved = self.logging(i_episode, scores_window, score, eps, mode, solved, agent, t)

            self.intermediate_save(save_every, i_episode, scores, agent, save_path, mode, save_prefix, render)

        self.save_and_plot(scores, agent, save_path, mode, save_prefix, render)

        te = pd.Timestamp.utcnow()
        logger.info(f"Elapsed Time: {pd.Timedelta(te - ts)}")
        return scores

    def logging(self, i_episode, scores_window, score, eps, mode, solved, agent, n_steps):
        self.agent_losses.append(float(agent.avg_loss))
        mean_loss = np.mean(self.agent_losses[:-min(len(self.agent_losses), 100)])
        log = f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}, Agent Loss: '\
              f'{mean_loss:.2e}, Last Score: {score:.2f} '\
              f'({n_steps} '\
              f'steps), '\
              f'eps: {eps:.2f}'
        print(log, end="")
        logger.debug(log)
        if i_episode % 100 == 0:
            logger.info(log)

        if self.threshold and np.mean(scores_window) >= self.threshold and mode == "train" and not solved:
            logger.warning(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score:'
                           f' {np.mean(scores_window):.2f}')
            solved = True
        return solved

    def intermediate_save(self, save_every, i_episode, scores, agent, save_path, mode, save_prefix, render):
        if save_every and mode == "train" and save_path and i_episode % save_every == 0:
            logger.info(f'\nSaving model to {save_path}')
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            if not render:
                plot_scores(scores, path=save_path, threshold=self.threshold, prefix=save_prefix)
                plot_scores(self.agent_losses, path=save_path, prefix=save_prefix + '_agent_loss')

    def save_and_plot(self, scores, agent, save_path, mode, save_prefix, render):
        if save_path and mode == "train":
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            if not render:
                plot_scores(scores, path=save_path, threshold=self.threshold, prefix=save_prefix)
                plot_scores(self.agent_losses, path=save_path, prefix=save_prefix+'_agent_loss')
        elif mode == "test":
            plot_scores(scores, path=".", threshold=self.threshold, prefix=save_prefix)

    def play(self, agent):
        state = self.reset()
        for _ in range(2000):
            action = agent.act(state)
            self.env.render()
            state, reward, done, _ = self.env.step(action)
            if done:
                break
        self.env.close()

    def process_action(self, actions):
        raise NotImplementedError