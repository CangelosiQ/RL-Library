"""
 Created by quentincangelosi at 17.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import logging

from rl_library.monitors.base_monitor import Monitor

logger = logging.getLogger("rllib.unity-monitor")


class UnityMonitor(Monitor):

    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

        # reset the environment
        env_info = env.reset(train_mode=True)[self.brain_name]
        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        self.action_size = self.brain.vector_action_space_size
        print('Size of each action:', self.action_size)

        # examine the state space
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like:', states[0])

    def reset_env(self):
        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def env_step(self, action):
        env_info = self.env.step(self.process_action(action))[self.brain_name]  # send the action to the environment
        next_state = env_info.vector_observations # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished
        return next_state, reward, done, None

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
        return actions
