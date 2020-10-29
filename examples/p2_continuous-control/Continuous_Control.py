#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

from unityagents import UnityEnvironment
import numpy as np
import logging
import os
import pandas as pd
from pathlib import Path

from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.monitors import unity_monitor
from rl_library.agents import DQAgent

# ---------------------------------------------------------------------------------------------------
#  INPUTS
# ---------------------------------------------------------------------------------------------------
hidden_layer_sizes = [20, 15, 8]
mode = "train"  # "train" or "test"
save_path = f"DDPG_" + "_".join([str(sz) for sz in hidden_layer_sizes])
os.makedirs(save_path, exist_ok=True)
# ---------------------------------------------------------------------------------------------------

# Logger
logger = logging.getLogger()
handler = logging.FileHandler(f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log")
stream_handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addHandler(stream_handler)

path = Path(__file__).parent
# ------------------------------------------------------------
#  1. Initialization
# ------------------------------------------------------------
env = UnityEnvironment(file_name='./Reacher 2.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ------------------------------------------------------------
#  2. Training
# ------------------------------------------------------------
def post_process_action(actions):
    return np.clip(actions, -1, 1)


if mode == "train":
    agent = DDPGAgent(state_size=state_size, action_size=action_size,
                    #hidden_layer_sizes=hidden_layer_sizes,
                    #post_process_action=post_process_action,
                      )

    scores = unity_monitor.run(env, agent, brain_name, save_every=500, save_path=save_path)
    logger.info("Average Score last 100 episodes: {}".format(np.mean(scores[-100:])))

# Testing
else:
    agent = DDPGAgent.load(filepath=save_path, mode="test")
    scores = unity_monitor.run(env, agent, brain_name, n_episodes=10, length_episode=1e6, mode="test")
    logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")

# When finished, you can close the environment.
logger.info("Closing...")
env.close()

# env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
# states = env_info.vector_observations                  # get the current state (for each agent)
# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
#
#
