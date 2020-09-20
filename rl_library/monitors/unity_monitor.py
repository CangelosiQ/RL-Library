import pickle
from collections import deque
import numpy as np
import os
from rl_library.utils.visualization import plot_scores


def run(env, agent, brain_name, n_episodes=2000, length_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        save_path:str = None, save_every: int = None, reload_path:str = None, mode: str = "train"):

    # ------------------------------------------------------------
    #  1. Initialization
    # ------------------------------------------------------------
    # reset the environment
    scores = []  # list containing scores from each episode
    if reload_path is not None and os.path.exists(reload_path + "/checkpoint.pth"):
        print(f"Reloading session from {reload_path}")
        agent.load(filepath=reload_path)
        with open(reload_path + "/scores.pickle", "rb") as f:
            scores = pickle.load(f)

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=mode=="train")[brain_name]
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        actions_counter = {i: 0 for i in range(agent.action_size)}
        for t in range(length_episode):
            action = agent.act(state, eps)
            actions_counter[action] += 1
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}, Actions: {}, Last Score: {:.2f}'.format(i_episode,
                                                                                            np.mean(scores_window),
                                                                                            actions_counter,
                                                                                            score
                                                                                            )) #, end=""
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break

        if save_every and save_path and i_episode % save_every == 0:
            print(f'\nSaving model to {save_path}')
            agent.save(filepath=save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(save_path + "/scores.pickle", "wb") as f:
                pickle.dump(scores, f)
            rolling_mean = plot_scores(scores)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        agent.save(filepath=save_path)
        with open(save_path + "/scores.pickle", "wb") as f:
            pickle.dump(scores, f)

    return scores