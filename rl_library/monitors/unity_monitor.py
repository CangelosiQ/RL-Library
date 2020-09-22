import pickle
from collections import deque
import numpy as np
import os
import logging
logger = logging.getLogger()
import pandas as pd
from rl_library.utils.visualization import plot_scores


def run(env, agent, brain_name, n_episodes=2000, length_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        save_path:str = None, save_every: int = None, reload_path:str = None, mode: str = "train"):
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
        eps_start = 0
        eps_decay = 0
        eps_end = 0
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=mode=="train")[brain_name]
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        # actions_counter = {i: 0 for i in range(agent.action_size)}
        for t in range(int(length_episode)):
            action = agent.act(state, eps)
            # actions_counter[action] += 1
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            if mode == "train":
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
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

        if save_every and mode == "train" and save_path and i_episode % save_every == 0:
            logger.info(f'\nSaving model to {save_path}')
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            agent.save(filepath=save_path)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            rolling_mean = plot_scores(scores, path=save_path, threshold=13)

    if save_path and mode == "train":
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        agent.save(filepath=save_path)
        with open(save_path + "/scores.pickle", "wb") as f:
            pickle.dump(scores, f)
        rolling_mean = plot_scores(scores, path=save_path, threshold=13)
    elif mode == "test":
        rolling_mean = plot_scores(scores, path=".", threshold=13)

    te = pd.Timestamp.utcnow()
    logger.info(f"Elapsed Time: {pd.Timedelta(te-ts)}")
    return scores