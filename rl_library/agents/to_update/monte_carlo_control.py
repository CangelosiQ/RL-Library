"""
 Created by quentincangelosi at 06.09.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import numpy as np
from collections import defaultdict
import sys


def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate episode
        episode = generate_episode(env)

        # Update N and return_sum
        cum_reward = 0
        for state, action, reward in reversed(episode):
            # Every visit
            returns_sum[state][action] += reward + gamma * cum_reward
            N[state][action] += 1
            cum_reward += reward

    # Update Q
    for state in returns_sum.keys():
        for action in range(env.action_space.n):
            Q[state][action] = returns_sum[state][action] / N[state][action]

    return Q


def mc_control(env, num_episodes, generate_episode, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    policy = {}
    epsilon = 1
    total_returns = []
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #        print(f"\nEpisode: {i_episode}, Epsilon: {epsilon}")
        #        print(f"Policy: {policy}")
        #        print(f"Q: {Q}")

        # Generate episode
        episode = generate_episode(env, policy, eps=epsilon)
        episode_length = len(episode)
        states, actions, rewards = zip(*episode)
        total_returns.append(sum(rewards))
        discount_rates = np.array([gamma ** i for i in range(len(states))])

        # Update N and return_sum
        for i, (state, action) in enumerate(zip(states, actions)):
            # Every visit
            if state not in Q:
                Q[state][action] = 0
            G = sum(rewards[i:] * discount_rates[:(episode_length - i)])
            Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])

            # Update policy
            policy[state] = np.argmax(Q[state])

        epsilon = max(1 - i_episode / num_episodes, 0.25)

    return policy, Q, total_returns


class MonteCarlo():

    def __init__(self, generate_episode):
        self.generate_episode = generate_episode

    def montecarlo(self, env, Q, total_returns, policy, gamma, alpha, epsilon):
        # Generate episode
        episode = self.generate_episode(env, policy, eps=epsilon)
        episode_length = len(episode)
        states, actions, rewards = zip(*episode)
        total_returns.append(sum(rewards))
        discount_rates = np.array([gamma ** i for i in range(len(states))])

        # Update N and return_sum
        for i, (state, action) in enumerate(zip(states, actions)):
            # Every visit
            if state not in Q:
                Q[state][action] = 0
            G = sum(rewards[i:] * discount_rates[:(episode_length - i)])
            Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])

            # Update policy
            policy[state] = np.argmax(Q[state])

        return Q, policy, total_returns