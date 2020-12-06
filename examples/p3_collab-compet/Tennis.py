#!/usr/bin/env python
# coding: utf-8
"""
---------------------------------------
    Project 3: Tennis
---------------------------------------
In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
program.


TODO:
    -

"""

# ---------------------------------------------------------------------------------------------------
#  External Dependencies
# ---------------------------------------------------------------------------------------------------
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project
# instructions to double-check that you have installed
# [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
# and [NumPy](http://www.numpy.org/).
import numpy as np
import logging
import os
import pandas as pd
import torch
import torch.nn.functional as F

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from unityagents import UnityEnvironment
from rl_library.agents.maddpg_agent import MADDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.unity_monitor import UnityMonitor


def main(seed=seed):
    # ---------------------------------------------------------------------------------------------------
    #  Logger
    # ---------------------------------------------------------------------------------------------------
    save_path = f"./results/Tennis_DDPG_{pd.Timestamp.utcnow().value}"
    os.makedirs(save_path, exist_ok=True)

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

    handler = logging.FileHandler(f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # ---------------------------------------------------------------------------------------------------
    #  Inputs
    # ---------------------------------------------------------------------------------------------------
    n_episodes = 2500
    config = dict(
        # Environment parameters
        env_name="Tennis",
        n_episodes=n_episodes,
        length_episode=1500,
        save_every=500,
        save_path=save_path,
        mode="train",  # "train" or "test"
        evaluate_every=5000,  # Number of training episodes before 1 evaluation episode
        eps_decay=1,  # Epsilon decay rate

        # Agent Parameters
        agent="DDPG",
        hidden_layers_actor=(256, 128),  # (50, 50, 15),  # (200, 150),  #
        hidden_layers_critic_body=(256,),  # (50, 50,),  #
        hidden_layers_critic_head=(128,),  # (50,),   # (300,)
        func_critic_body="F.leaky_relu",  #
        func_critic_head="F.leaky_relu",  #
        func_actor_body="F.leaky_relu",  #
        # lr_scheduler=None,
        lr_scheduler={'scheduler_type': "exp", #"multistep",  # "step", "exp" or "decay", "multistep"
                      'gamma': 0.99999,  #0.75,
                      'step_size': 1,
                      # 'milestones': [30 * 1000 * i for i in range(1, 6)],
                      'max_epochs': n_episodes},

        TAU=1e-3,  # for soft update of target parameters
        BUFFER_SIZE=int(3e4),  # replay buffer size
        BATCH_SIZE=128,  # minibatch size
        GAMMA=0.99,  # discount factor
        LR_ACTOR=1e-4,  # learning rate of the actor
        LR_CRITIC=1e-4,  # learning rate of the critic
        WEIGHT_DECAY=0,  # L2 weight decay
        UPDATE_EVERY=1,  # Number of actions before making a learning step
        N_CONSECUTIVE_LEARNING_STEPS=2,
        action_noise="OU",  #
        action_noise_scale=1,
        weights_noise=None,  #
        state_normalizer="BatchNorm",  # "RunningMeanStd" or "BatchNorm"
        warmup=1e3,  # Number of random actions to start with as a warm-up
        start_time=str(pd.Timestamp.utcnow()),
        random_seed=seed,
        threshold=0.5
    )
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
    env_info = env.reset(train_mode=True)[brain_name]

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
                                func=torch.tanh, seed=seed)
    # Critic model
    critic = DeepNeuralNetHeadCritic(action_size*num_agents,
                                     SimpleNeuralNetBody(state_size*num_agents, config["hidden_layers_critic_body"],
                                                         func=eval(config["func_critic_body"]), seed=seed),
                                     hidden_layers_sizes=config["hidden_layers_critic_head"],
                                     func=eval(config["func_critic_head"]),
                                     end_func=None, seed=seed)

    # MADDPG Agent
    agent = MADDPGAgent(state_size=state_size, action_size=action_size,
                        model_actor=actor, model_critic=critic,
                        action_space_low=-1, action_space_high=1,
                        config=config,
                        )

    if config["mode"] == "train":
        # Training
        start = pd.Timestamp.utcnow()
        scores = monitor.run(agent)
        logger.info("Average Score last 100 episodes: {}".format(np.mean(scores[-100:])))
        elapsed_time = pd.Timedelta(pd.Timestamp.utcnow() - start)
        logger.info(f"Elapsed Time: {elapsed_time}")

    # ------------------------------------------------------------
    #  3. Testing
    # ------------------------------------------------------------
    else:
        agent.load(filepath=config['save_path'], mode="test")
        scores = monitor.run(agent)
        logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")
        config["test_scores"] = scores
        config["best_test_score"] = max(scores)
        config["avg_test_score"] = np.mean(scores)

    # When finished, you can close the environment.
    logger.info("Closing...")
    env.close()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
