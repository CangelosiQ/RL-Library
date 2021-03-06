"""
 Created by quentincangelosi at 03.11.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import json
from collections import deque
from copy import copy

import logging
import numpy as np
import pandas as pd
import os, pickle
import torch

from rl_library.utils.utils import NpEncoder
from rl_library.utils.visualization import plot_scores

logger = logging.getLogger("rllib.monitor")


class Monitor:

    def __init__(self, config: dict):
        self.config = config
        self.env_name = config["env_name"]
        self.env = config.get("env")
        self.seed = config.get("random_seed", 42)

        self.threshold = config.get("threshold")
        self.agent_losses = []
        self.n_episodes = config.get("n_episodes", 2000)
        self.length_episode = config.get("length_episode", 500)
        self.mode = config.get("mode", "train")
        self.reload_path = config.get("reload_path")
        self.save_every = config.get("save_every", 50)

        self.save_path = config.get('save_path')
        self.render = config.get("render", False)

        if self.mode != "train":
            self.eps_start = 0
            self.eps_decay = 0
            self.eps_end = 0
        else:
            # Epsilon Parameter for balancing Exploration and exploitation
            self.eps_start = 1.0
            self.eps_end = 0.01
            self.eps_decay = config.get("eps_decay", 0.995)
        # If evaluate_every is not None, then the balancing Exploration and exploitation
        self.evaluate_every = config.get("evaluate_every")
        self.start = pd.Timestamp.utcnow()
        self.n_agents = config.get("n_agents", 1)
        self.func_scoring = np.max
        self.best_score_rolling100 = -np.inf

    def reset(self):
        state = self.reset_env()
        last_actions = deque(maxlen=100)
        last_states = deque(maxlen=100)
        scores = np.zeros(self.n_agents)
        list_actions = []
        list_rewards = []
        list_states = [state]
        return state, list_states, list_rewards, list_actions, scores, last_states, last_actions

    def reset_env(self):
        raise NotImplementedError

    def env_step(self, action):
        raise NotImplementedError

    def run(self, agent):
        save_prefix = f'{self.env_name}_{agent.__class__.__name__}'
        ts = pd.Timestamp.utcnow()
        # ------------------------------------------------------------
        #  1. Initialization
        # ------------------------------------------------------------
        # reset the environment
        scores_history = []  # list containing scores from each episode
        solved = False
        if self.reload_path is not None and os.path.exists(self.reload_path + "/checkpoint.pth"):
            logger.info(f"Reloading session from {self.reload_path}")
            with open(self.reload_path + "/scores.pickle", "rb") as f:
                scores_history = pickle.load(f)

        scores_window = deque(maxlen=100)  # last 100 scores
        self.evaluation_scores = []
        eps = self.eps_start  # initialize epsilon
        t = None
        for i_episode in range(1, self.n_episodes + 1):
            # list_states, list_rewards, list_actions are in the case where agents are updated once at the end of an
            # episode
            state, list_states, list_rewards, list_actions, scores, last_states, last_actions = self.reset()
            agent.reset()  # Reset Noise Random Process

            # Turn ON Evaluation Episode
            if self.evaluate_every is not None and i_episode % self.evaluate_every == 0:
                logger.warning("Evaluation episode ACTIVATED")
                agent.training = False  # TODO Not working for MADDPG use function instead

            for t in range(int(self.length_episode)):
                action = agent.act(state, eps=eps * int(self.mode == "train"))

                # print(f"\rAction: {action} Processed: {self.process_action(action)}", end="")
                next_state, reward, done, _ = self.env_step(self.process_action(action))  # send the action to the
                # last_actions.append(action)
                # last_states.append(state)
                # if len(last_actions) == 100:
                #     diff = np.sum(np.std(last_actions, 0))
                #     diff_state = np.sum(np.std(last_states, 0))
                #     if diff < 0.4 and diff_state < 0.2:
                #         print(f"Episode {i_episode}: AGENT STUCK! diff={diff}, diff_state={diff_state}, {action}")
                #         reward -= 0.01
                # break

                # environment
                if self.render:
                    self.env.render()

                if self.mode == "train" and agent.step_every_action:
                    agent.step(state, action, reward, next_state, done)

                # We are going to update the agent only after the end of the episode
                elif self.mode == "train":
                    list_actions.append(action)
                    list_states.append(next_state)
                    list_rewards.append(reward)

                state = next_state
                scores += reward

                done = [done, ] if type(done) is not list else done
                if any(done):
                    break

            # Turn OFF
            if self.evaluate_every is not None and i_episode % self.evaluate_every == 0:
                self.evaluation_scores.append(np.mean(scores))  # save most recent score
                logger.warning(f"Average Evaluation Score: {np.mean(self.evaluation_scores):.2f}, Last Score: "
                               f"{np.mean(scores)}")
                agent.training = True
                logger.warning("Evaluation episode DEACTIVATED")
            else:
                if self.mode == "train" and not agent.step_every_action:
                    agent.step(list_states, list_actions, list_rewards)

            scores_window.append(self.func_scoring(scores))  # save most recent score
            scores_history.append(self.func_scoring(scores))  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            solved = self.logging(i_episode, scores_window, scores, eps, solved, agent, t)

            self.intermediate_save(i_episode, scores_history, agent, save_prefix)

        self.save_and_plot(scores_history, agent, save_prefix)

        te = pd.Timestamp.utcnow()
        logger.info(f"Elapsed Time: {pd.Timedelta(te - ts)}")
        return scores_history

    def logging(self, i_episode, scores_window, score, eps, solved, agent, n_steps):
        self.agent_losses.append(agent.avg_loss)

        if len(self.agent_losses) <= 100:
            mean_loss = np.mean(np.array(self.agent_losses[:len(self.agent_losses)]), axis=0)
        else:
            mean_loss = np.mean(np.array(self.agent_losses[:-100]), axis=0)

        log = f'Episode {i_episode}    Average Score: {np.mean(scores_window):.2e}, Agent Loss: ' \
              f'{mean_loss}, Last Score: avg={np.mean(score):.2e}, min={min(score):.2e}, max={max(score):.2e} ' \
              f'({n_steps} ' \
              f'steps), ' \
              f'eps: {eps:.2f}'
        if i_episode % 250 == 0:
            logger.info(log)
            logger.info(str(agent))
        if i_episode % 25 == 0:
            logger.info(log)
        else:
            logger.debug(log)

        if self.threshold and np.mean(scores_window) >= self.threshold and self.mode == "train" and not solved:
            logger.warning(f'\n========>>> Environment solved in {i_episode} episodes!\tAverage Score:'
                           f' {np.mean(scores_window):.2f}')
            solved = True
        return solved

    def intermediate_save(self, i_episode, scores, agent, save_prefix):
        if self.save_every and self.mode == "train" and self.save_path and i_episode % self.save_every == 0:
            self._save_training(agent, scores, save_prefix, i_episode)

        # Save best model
        elif self.mode == "train" and self.save_path and len(scores)>100 and self.threshold:
            new_score = np.mean(scores[-100:])
            if new_score > self.threshold and new_score > self.best_score_rolling100:
                self.best_score_rolling100 = new_score
                logger.warning(f'\n========>>> New Best Score {i_episode} episodes!\tAverage '
                               f'Score: {new_score:.2f}')
                self._save_training(agent, scores, save_prefix, save_path=self.save_path+"/best")

    def save_and_plot(self, scores, agent, save_prefix):
        if self.save_path and self.mode == "train":
            self._save_training(agent, scores, save_prefix)
        elif self.mode == "test":
            if len(np.array(scores).shape) > 1:
                scores = np.mean(scores, axis=1)
            plot_scores(list(scores), path=".", threshold=self.threshold, prefix=save_prefix)

    def _save_training(self, agent, scores, save_prefix, i_episode=None, save_path=None):
        if save_path is None:
            save_path = self.save_path
        logger.info(f'Saving model to {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Save agent
        agent.save(filepath=save_path)

        # Save scores
        with open(save_path + "/scores.pickle", "wb") as f:
            pickle.dump(scores, f)

        # Plot scores
        if not self.render:
            self.plots(scores, save_prefix)

        # Save config
        self.save_config(scores, i_episode)

    def plots(self, scores, save_prefix):
        if len(np.array(scores).shape) > 1:
            scores = list(np.mean(scores, axis=1))

        plot_scores(scores, path=self.save_path, threshold=self.threshold,
                    prefix=save_prefix)

        if len(self.evaluation_scores) > 0:
            plot_scores(self.evaluation_scores, path=self.save_path, threshold=self.threshold,
                        prefix=save_prefix + '_evaluation')
        plot_scores(np.array(self.agent_losses).transpose(), path=self.save_path, prefix=save_prefix + '_agent_loss',
                    log=True)

    def save_config(self, scores, i_episode=None):
        self.config["current_episode"] = i_episode
        self.config["training_scores"] = scores
        if len(np.array(scores).shape) > 1:
            scores = np.mean(np.array(scores), axis=1)
        self.config["best_training_score"] = np.max(scores)
        self.config["avg_training_score"] = np.mean(scores)
        if len(self.evaluation_scores) > 0:
            self.config["eval_scores"] = self.evaluation_scores
            self.config["best_eval_score"] = np.max(self.evaluation_scores)
            self.config["avg_eval_score"] = np.mean(self.evaluation_scores)

        if len(scores) > 50:
            self.config["last_50_score"] = np.mean(scores[:-50])

        self.config["elapsed_time"] = pd.Timedelta(pd.Timestamp.utcnow() - self.start).total_seconds()
        with open(f"./{self.config['save_path']}/config.json", "w") as f:
            json.dump(self.config, f, cls=NpEncoder)

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
