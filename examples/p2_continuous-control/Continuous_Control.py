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

TODO:
    DONE - Proper Book-Keeeping
    DONE - Raise number of episode to 2000 to approximately match the DDPG 2.5 million steps in the results table
    DONE - split agent scores between actor and critic to follow better what is hapening there
    DONE - instead of epsilon decay, split into evaluation and exploration phases (eg. evaluate for 1 episode every 50
    episodes?)
    DONE - scale=1 for OUNoise
    - interm save confi
    - slowly decaying the learning rate as the model approaches an optima
    - activate batch normalization
    - do running mean normalization
    - parameter noise
    - Try even more extreme Discount factor
    - Double DDDPG? Rainbow DDPG?
    - change noise (try random noise, adaptive noise)
    - agent avg_loss only represent a few actions ?


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
from pathlib import Path
import torch
import json
import sys
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors import unity_monitor
from rl_library.monitors.unity_monitor import UnityMonitor


def main(discount_factor=0.99, weight_decay=0.0001, batch_size=64):
    from unityagents import UnityEnvironment
    # ---------------------------------------------------------------------------------------------------
    #  Logger
    # ---------------------------------------------------------------------------------------------------
    path = Path(__file__).parent
    save_path = f"DDPG_{pd.Timestamp.utcnow().value}"
    os.makedirs(save_path, exist_ok=True)

    # logging.basicConfig(filename=f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log",
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                     level=logging.INFO)
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

    handler = logging.FileHandler(f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    # logger.addHandler(stream_handler)
    logger.addHandler(handler)
    # ---------------------------------------------------------------------------------------------------
    #  Inputs
    # ---------------------------------------------------------------------------------------------------
    n_episodes = 500
    config = dict(
        # Environment parameters
        env_name="Reacher 2",
        n_episodes=n_episodes,
        length_episode=1500,
        save_every=100,
        save_path=save_path,
        mode="train",  # "train" or "test"
        evaluate_every=50,  # Number of training episodes before 1 evaluation episode
        eps_decay=0.99,  # Epsilon decay rate

        # Agent Parameters
        agent="DDPG",
        hidden_layers_actor=(100, 30,),  #
        hidden_layers_critic_body=(100,),  #
        hidden_layers_critic_head=(30,),  #
        func_critic_body="F.relu",  #
        func_critic_head="F.relu",  #
        func_actor_body="F.relu",  #
        lr_scheduler={'scheduler_type': "exp",  # "step", "exp" or "decay"
                      'gamma': 0.99999,
                      'step_size': 1,
                      'max_epochs': n_episodes},

        TAU=1e-3,  # for soft update of target parameters
        BUFFER_SIZE=int(1e5),  # replay buffer size
        BATCH_SIZE=batch_size,  # minibatch size
        GAMMA=discount_factor,  # discount factor
        LR_ACTOR=5e-3,  # learning rate of the actor
        LR_CRITIC=5e-3,  # learning rate of the critic
        WEIGHT_DECAY=weight_decay,  # L2 weight decay
        UPDATE_EVERY=1,  # Number of actions before making a learning step
        action_noise="OU",  #
        weights_noise=None,  #
        batch_normalization=None,  #
        warmup=0,  # Number of random actions to start with as a warm-up
        start_time=str(pd.Timestamp.utcnow()),
    )

    # ------------------------------------------------------------
    #  1. Initialization
    # ------------------------------------------------------------
    # 1. Start the Environment

    env = UnityEnvironment(file_name=f'./Reacher_Linux/Reacher.x86_64')  # Linux
    #env = UnityEnvironment(file_name=f'./{config["save_path"]}')  # mac OS

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
    config.update(dict(action_size=action_size, state_size=state_size))

    # ------------------------------------------------------------
    #  2. Training
    # ------------------------------------------------------------
    if config["mode"] == "train":
        # Actor model
        actor = SimpleNeuralNetHead(action_size, SimpleNeuralNetBody(state_size, config["hidden_layers_actor"]),
                                    func=torch.tanh)
        # Critic model
        critic = DeepNeuralNetHeadCritic(action_size,
                                         SimpleNeuralNetBody(state_size, config["hidden_layers_critic_body"],
                                                             func=eval(config["func_critic_body"])),
                                         hidden_layers_sizes=config["hidden_layers_critic_head"],
                                         func=eval(config["func_critic_head"]),
                                         end_func=None)

        # DDPG Agent
        agent = DDPGAgent(state_size=state_size, action_size=action_size,
                          model_actor=actor, model_critic=critic,
                          action_space_low=[-1, ] * action_size, action_space_high=[1, ] * action_size,
                          config=config,
                          )

        # Unity Monitor
        monitor = UnityMonitor(env=env, config=config)
        monitor.eps_decay = config["eps_decay"]

        # Training
        start = pd.Timestamp.utcnow()
        scores = monitor.run(agent)
        logger.info("Average Score last 100 episodes: {}".format(np.mean(scores[-100:])))
        elapsed_time = pd.Timedelta(pd.Timestamp.utcnow() - start).total_seconds()
        logger.info(f"Elapsed Time: {elapsed_time} seconds")
        config["training_time"] = elapsed_time
        config["training_scores"] = scores
        config["best_training_score"] = max(scores)
        config["avg_training_score"] = np.mean(scores)
        config["last_50_score"] = np.mean(scores[:-50])
    # ------------------------------------------------------------
    #  3. Testing
    # ------------------------------------------------------------
    else:
        agent = DDPGAgent.load(filepath=config['save_path'], mode="test")
        scores = unity_monitor.run(env, agent, brain_name, n_episodes=10, length_episode=1e6, mode="test")
        logger.info(f"Test Score over {len(scores)} episodes: {np.mean(scores)}")
        config["test_scores"] = scores
        config["best_test_score"] = max(scores)
        config["avg_test_score"] = np.mean(scores)

    with open(f"./{config['save_path']}/config.json", "w") as f:
        json.dump(config, f)

    # When finished, you can close the environment.
    logger.info("Closing...")
    env.close()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    skip_first = 0
    for batch_size in [256]:
        for weight_decay in [ 0.01, 0.0001, ]:
            for discount_factor in [ 0.9, 0.7, 0.8,]:
                if skip_first > 0:
                    skip_first -= 1
                    continue
                logger.warning("+="*90)
                logger.warning(f"  RUNNING SIMULATION WITH PARAMETERS discount_factor={discount_factor}, "
                               f"weight_decay={weight_decay}, batch_size={batch_size}")
                logger.warning("+="*90)
                main(discount_factor=discount_factor, weight_decay=weight_decay, batch_size=batch_size)
