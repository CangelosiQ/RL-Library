# # Navigation
# ---
#
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
#
# ### 1. Start the Environment
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).
from unityagents import UnityEnvironment
from rl_library.monitors import unity_monitor
from rl_library.agents import DQAgent

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86")

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
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
agent = DQAgent(state_size=state_size, action_size=action_size, hidden_layer_sizes=[round(state_size/2), round(state_size/2)])

score = unity_monitor.run(env, agent, brain_name, save_every=200, save_path=".")
print("Score: {}".format(score))

# When finished, you can close the environment.
print("Closing...")
env.close()
