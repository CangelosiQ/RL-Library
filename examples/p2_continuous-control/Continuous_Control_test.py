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
import json

import numpy as np
import logging
import os
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from unityagents import UnityEnvironment
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.unity_monitor import UnityMonitor


def main(seed=seed):
    # ---------------------------------------------------------------------------------------------------
    #  Logger
    # ---------------------------------------------------------------------------------------------------
    results_path = f"."

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

    handler = logging.FileHandler(f"{results_path}/logs_test_{pd.Timestamp.utcnow().value}.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # ---------------------------------------------------------------------------------------------------
    #  Inputs
    # ---------------------------------------------------------------------------------------------------
    with open(f"{results_path}/config.json", "r") as f:
        config = json.load(f)
    #save_path = results_path,
    config["mode"] = "test"
    config["n_episodes"] = 10

    logger.warning("+=" * 90)
    logger.warning(f"  RUNNING SIMULATION WITH PARAMETERS config={config}")
    logger.warning("+=" * 90)

    # ------------------------------------------------------------
    #  1. Initialization
    # ------------------------------------------------------------
    # 1. Start the Environment

    # env = UnityEnvironment(file_name=f'./Reacher_Linux_2/Reacher.x86_64')  # Linux
    env = UnityEnvironment(file_name=f'./{config["env_name"]}')  # mac OS

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    config["n_agents"] = num_agents

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    config.update(dict(action_size=action_size, state_size=state_size))

    # ------------------------------------------------------------
    #  2. Training
    # ------------------------------------------------------------
    # Unity Monitor
    monitor = UnityMonitor(env=env, config=config)


    # Actor model
    seed = 0
    actor = SimpleNeuralNetHead(action_size,
                                SimpleNeuralNetBody(state_size, config["hidden_layers_actor"], seed=seed),
                                func=F.tanh, seed=seed)

    # Critic model
    critic = DeepNeuralNetHeadCritic(action_size,
                                     SimpleNeuralNetBody(state_size, config["hidden_layers_critic_body"],
                                                         func=eval(config["func_critic_body"]), seed=seed),
                                     hidden_layers_sizes=config["hidden_layers_critic_head"],
                                     func=eval(config["func_critic_head"]),
                                     end_func=None, seed=seed)

    # DDPG Agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size,
                      model_actor=actor, model_critic=critic,
                      action_space_low=-1, action_space_high=1,
                      config=config,
                      )
    agent.load(results_path)

    # ------------------------------------------------------------
    #  3. Testing
    # ------------------------------------------------------------
    start = pd.Timestamp.utcnow()
    scores = monitor.run(agent)
    elapsed_time = pd.Timedelta(pd.Timestamp.utcnow() - start).total_seconds()
    logger.info(f"Elapsed Time: {elapsed_time} seconds")

    logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")
    config["test_scores"] = scores
    config["best_test_score"] = np.max(np.mean(np.array(scores), axis=1))
    config["avg_test_score"] = np.mean(np.mean(np.array(scores), axis=1))

    # When finished, you can close the environment.
    logger.info("Closing...")
    env.close()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
