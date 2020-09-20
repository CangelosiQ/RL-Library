"""
 Created by quentincangelosi at 06.09.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class ControlProblem(object):

    def __init__(self, control_algorithm, plot_values=None):
        self.control_algorithm = control_algorithm
        self.plot_values = plot_values

    def __str__(self):
        return "ControlProblem()"

    def run_control(self, env, num_episodes, alpha, gamma=1.0):
        nA = env.action_space.n
        nS = env.observation_space.n
        # initialize empty dictionary of arrays
        epsilon = 1
        Q = defaultdict(lambda: np.zeros(nA))
        policy = {}
        total_returns = []
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                # print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                print(f"\nEpisode: {i_episode}/{num_episodes}, Epsilon: {epsilon}")
                print(f"Policy:")
                self.print_policy_cliff(policy)
                #self.heatmap_Q(nS, nA, Q)
                self.plot_state_value_function(Q)
                sys.stdout.flush()

            Q, policy, total_returns = self.control_algorithm(env, Q, total_returns,
                                                                   policy, gamma, alpha, epsilon)

            # epsilon = max(1 / i_episode, 0.1)
            epsilon = 1 / i_episode

        return policy, Q, total_returns


    @staticmethod
    def heatmap_Q(nS, nA, Q):
        plt.figure(figsize=(10,10))
        data = [Q[state] if state in Q else np.zeros(nA) for state in range(nS)]
        # plt.imshow(data, cmap='viridis')
        ax = sns.heatmap(data, linewidth=0.5)
        plt.show()

    def plot_state_value_function(self, Q):
        # plot the estimated optimal state-value function
        if self.plot_values:
            V_sarsa = ([np.max(Q[key]) if key in Q else 0 for key in np.arange(48)])
            self.plot_values(V_sarsa)
            plt.show()

    @staticmethod
    def print_policy_cliff(policy):

        data = np.zeros(48)
        for state in policy.keys():
            data[state] = policy[state]
        data = data.reshape(4, 12)

        actions = {0: "U",
                   1: "R",
                   2: "D",
                   3: "L"}
        line = '-'*12
        print(line)
        for col in data:
            for row in col:
                print(actions[int(row)], end=' | ')
            print()
            print(line)