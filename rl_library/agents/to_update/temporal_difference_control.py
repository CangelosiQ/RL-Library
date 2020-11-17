"""
 Created by quentincangelosi at 06.09.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import numpy as np


class TemporalDifferences():

    @staticmethod
    def greedy_policy(state, policy, nA, eps=0.1):
        if state in policy:
            probs = [eps / nA for i in range(nA)]
            probs[policy[state]] += 1 - eps
        else:
            probs = [1 / nA for i in range(nA)]
        action = np.random.choice(np.arange(nA), p=probs)
        return action

    @staticmethod
    def get_stochastic_policy(policy, state, nA, eps):
        if state in policy:
            probs = [eps / nA for i in range(nA)]
            probs[policy[state]] += 1 - eps
        else:
            probs = [1 / nA for i in range(nA)]
        return probs

    @staticmethod
    def make_greedy_step(env, state, policy, eps=0.1):
        nA = env.action_space.n
        action = TemporalDifferences.greedy_policy(state, policy, nA, eps)
        next_state, reward, done, info = env.step(action)
        return state, action, reward, next_state, done

    @staticmethod
    def sarsa0(env, Q, total_returns, policy, gamma, alpha, epsilon):
        cum_reward = 0
        state = env.reset()
        nA = env.action_space.n

        # Make first step
        state, action, reward, next_state, done = TemporalDifferences.make_greedy_step(env, state, policy, epsilon)
        cum_reward += reward
        # print(f"State: {state}, Action: {action}, Reward: {reward} (cum: {cum_reward})")
        count = 0
        while count < 1000:
            count += 1
            if done:
                if state not in Q:
                    Q[state][action] = 0
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break

            # SARSA 0 need next action
            next_action = TemporalDifferences.greedy_policy(next_state, policy, nA, epsilon)

            # Update Q
            # Every visit
            if state not in Q:
                Q[state][action] = 0

            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            # Update policy
            policy[state] = np.argmax(Q[state])

            # Make a step with next action
            state = next_state
            action = next_action
            next_state, reward, done, info = env.step(action)
            cum_reward += reward
            #print(f"State: {state}, Action: {action}, Reward: {reward} (cum: {cum_reward})")

        # print(f"Total Steps in Episode: {count}, Done: {done}")
        total_returns.append(cum_reward)
        return Q, policy, total_returns

    @staticmethod
    def qlearning(env, Q, total_returns, policy, gamma, alpha, epsilon):
        """
            Also known as sarsamax
        """
        cum_reward = 0
        state = env.reset()
        nA = env.action_space.n

        # Make first step
        # print(f"State: {state}, Action: {action}, Reward: {reward} (cum: {cum_reward})")
        count = 0
        while count < 1000:
            count += 1

            # Make a step
            state, action, reward, next_state, done = TemporalDifferences.make_greedy_step(env, state, policy, epsilon)
            cum_reward += reward

            if done:
                if state not in Q:
                    Q[state][action] = 0
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break

            # SARSA Max takes best action
            best_action = policy[next_state] if next_state in policy else np.random.choice(nA)

            # Update Q
            # Every visit
            if state not in Q:
                Q[state][action] = 0

            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * Q[next_state][best_action] - Q[state][action])

            # Update policy
            policy[state] = np.argmax(Q[state])

            state = next_state

        # print(f"Total Steps in Episode: {count}, Done: {done}")
        total_returns.append(cum_reward)
        return Q, policy, total_returns

    @staticmethod
    def expected_sarsa(env, Q, total_returns, policy, gamma, alpha, epsilon):
        """
            Also known as sarsamax
        """
        cum_reward = 0
        state = env.reset()
        nA = env.action_space.n

        # Make first step
        # print(f"State: {state}, Action: {action}, Reward: {reward} (cum: {cum_reward})")
        count = 0
        while count < 1000:
            count += 1

            # Make a step
            state, action, reward, next_state, done = TemporalDifferences.make_greedy_step(env, state, policy, epsilon)
            cum_reward += reward

            if done:
                if state not in Q:
                    Q[state][action] = 0
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break

            # SARSA Max takes best action
            stochastic_policy = TemporalDifferences.get_stochastic_policy(policy, state, nA, epsilon)
            expected_return_next_state = np.dot(Q[next_state] , stochastic_policy) if next_state in Q else 0

            # Update Q
            # Every visit
            if state not in Q:
                Q[state][action] = 0

            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * expected_return_next_state - Q[state][action])

            # Update policy
            policy[state] = np.argmax(Q[state])

            state = next_state

        # print(f"Total Steps in Episode: {count}, Done: {done}")
        total_returns.append(cum_reward)
        return Q, policy, total_returns