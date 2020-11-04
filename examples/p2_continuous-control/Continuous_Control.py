#!/usr/bin/env python
# coding: utf-8
"""
---------------------------------------
    Project 2: Continuous Control
---------------------------------------
In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
program.

Notes:
    - Discount factor seem to have good influence, the agent should probably aim at the ball anytime so every action
    is important. Present actions even more than future because the best time to follow the ball is always NOW. Of
    course when the agent is far away it needs to know which suite of actions will get him back to the ball direction.

"""

# ---------------------------------------------------------------------------------------------------
#  External Dependencies
# ---------------------------------------------------------------------------------------------------
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project
# instructions to double-check that you have installed
# [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
# and [NumPy](http://www.numpy.org/).
from unityagents import UnityEnvironment
import numpy as np
import logging
import os
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors import unity_monitor
from rl_library.monitors.unity_monitor import UnityMonitor

# ---------------------------------------------------------------------------------------------------
#  Inputs
# ---------------------------------------------------------------------------------------------------
hidden_layer_sizes = [20, 15, 8]
mode = "train"  # "train" or "test"
path = Path(__file__).parent
save_path = f"DDPG_" + "_".join([str(sz) for sz in hidden_layer_sizes])
os.makedirs(save_path, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------------------------------
logger = logging.getLogger()
handler = logging.FileHandler(f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log")
stream_handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addHandler(stream_handler)

# ------------------------------------------------------------
#  1. Initialization
# ------------------------------------------------------------
# 1. Start the Environment
env_name = "Reacher 2"
env = UnityEnvironment(file_name=f'./{env_name}.app')

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
if mode == "train":
    # Actor model
    actor = SimpleNeuralNetHead(action_size, SimpleNeuralNetBody(state_size, (100,)),
                                func=F.tanh)
    # Critic model
    critic = DeepNeuralNetHeadCritic(action_size, SimpleNeuralNetBody(state_size, (100,),
                                                                      func=F.relu),
                                     hidden_layers_sizes=(50,),
                                     func=F.relu,
                                     end_func=None)

    # DDPG Agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size,
                      model_actor=actor, model_critic=critic, action_space_low=[-1, ] * action_size, action_space_high=
                      [1, ] * action_size, config={"warmup": 1e4},
                      hyper_parameters=dict(TAU=1e-3,  # for soft update of target parameters
                                            BUFFER_SIZE=int(1e5),  # replay buffer size
                                            BATCH_SIZE=128,  # minibatch size
                                            GAMMA=0.6,  # discount factor
                                            LR_ACTOR=1e-3,  # learning rate of the actor
                                            LR_CRITIC=1e-3,  # learning rate of the critic
                                            WEIGHT_DECAY=0.001,  # L2 weight decay
                                            UPDATE_EVERY=1,
                                            ))

    # Unity Monitor
    monitor = UnityMonitor(env_name=env_name, env=env)
    monitor.eps_decay = 0.99  # Epsilon decay rate

    # Training
    scores = monitor.run(agent, n_episodes=200, length_episode=1000, mode="train",
                         reload_path=None, save_every=100,
                         save_path=save_path)
    logger.info("Average Score last 100 episodes: {}".format(np.mean(scores[-100:])))

# ------------------------------------------------------------
#  3. Testing
# ------------------------------------------------------------
else:
    agent = DDPGAgent.load(filepath=save_path, mode="test")
    scores = unity_monitor.run(env, agent, brain_name, n_episodes=10, length_episode=1e6, mode="test")
    logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")

# When finished, you can close the environment.
logger.info("Closing...")
env.close()
