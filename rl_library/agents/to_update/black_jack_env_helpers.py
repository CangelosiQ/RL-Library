"""
 Created by quentincangelosi at 06.09.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import numpy as np


def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def generate_episode_eps_greedy(bj_env, policy, eps=0.1):
    episode = []
    state = bj_env.reset()
    nA = bj_env.action_space.n
    count = 0
    while count < 1000:
        count += 1
        # Epsilon Greedy policy
        if state in policy:
            probs = [eps/nA for i in range(nA)]
            probs[policy[state]] += 1-eps
        else:
            probs = [1/nA for i in range(nA)]
        action = np.random.choice(np.arange(nA), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


