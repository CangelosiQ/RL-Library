# # Navigation
# ---
#
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
#
# ### 1. Start the Environment
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).
from pathlib import Path

from unityagents import UnityEnvironment
import numpy as np
import logging
import os
import pandas as pd

from rl_library.monitors import unity_monitor
from rl_library.agents import DQAgent

# ---------------------------------------------------------------------------------------------------
#  INPUTS
# ---------------------------------------------------------------------------------------------------
hidden_layer_sizes = None # [20, 15, 8]
options = []     # ["double-q-learning", "prioritized-replay"]
mode = "train"                      # "train" or "test"
# save_path = f"DQN_" + "_".join([str(sz) for sz in hidden_layer_sizes])
save_path = f"DQN_simple"
os.makedirs(save_path, exist_ok=True)
# ---------------------------------------------------------------------------------------------------

# Logger
logger = logging.getLogger()
handler = logging.FileHandler(f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log")
stream_handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addHandler(stream_handler)

path = Path(__file__).parent
env = UnityEnvironment(file_name=f"{path}/Banana.app")  # "./Banana_Linux/Banana.x86"

# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
#
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana.
#
# Run the code cell below to print some information about the environment.

# ------------------------------------------------------------
#  1. Initialization
# ------------------------------------------------------------
# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
logger.info(f'Number of agents: {len(env_info.agents)}')

# number of actions
action_size = brain.vector_action_space_size
logger.info(f'Number of actions: {action_size}')

# examine the state space
state = env_info.vector_observations[0]
logger.info(f'States look like: {state}')
state_size = len(state)
logger.info(f'States have length: {state_size}')


# Training
if mode == "train":
    agent = DQAgent(state_size=state_size, action_size=action_size,
                    hidden_layer_sizes=hidden_layer_sizes, options=options)

    scores = unity_monitor.run(env, agent, brain_name, save_every=500, save_path=save_path)
    logger.info("Average Score last 100 episodes: {}".format(np.mean(scores[-100:])))

# Testing
else:
    agent = DQAgent.load(filepath=save_path, mode="test")
    scores = unity_monitor.run(env, agent, brain_name, n_episodes=10, length_episode=1e6, mode="test")
    logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")

# When finished, you can close the environment.
logger.info("Closing...")
env.close()
