"""
 Created by quentincangelosi at 28.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
# ---------------------------------------------------------------------------------------------------
#  External Dependencies
# ---------------------------------------------------------------------------------------------------
import json
import torch.nn.functional as F
import torch
from gym.spaces import Box
import logging
import pandas as pd
import numpy as np

np.random.seed(42)
seed = torch.manual_seed(42)

# ---------------------------------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------------------------------
# logging.basicConfig(filename=f"{save_path}/logs_navigation_{pd.Timestamp.utcnow().value}.log",
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')

log_fn = f"../../logs/logs_{__file__.split('/')[-1].replace('.py', '')}_{pd.Timestamp.utcnow().value}.log"
handler = logging.FileHandler(log_fn)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
#
logger.addHandler(stream_handler)
logger.addHandler(handler)

# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.openai_gym_monitor import GymMonitor


def main(discount_factor=0.99, weight_decay=0.0001, batch_size=64):
    # ---------------------------------------------------------------------------------------------------
    #  Inputs
    # ---------------------------------------------------------------------------------------------------
    n_episodes = 1000
    config = dict(
        # Environment parameters
        env_name='BipedalWalker-v3',
        n_episodes=n_episodes,
        length_episode=1500,
        save_every=100,
        save_path=f"./DDPG_Bipedal_{pd.Timestamp.utcnow().value}",
        mode="train",  # "train" or "test"
        evaluate_every=50,  # Number of training episodes before 1 evaluation episode
        eps_decay=0.9999,  # Epsilon decay rate
        render=True,

        # Agent Parameters
        agent="DDPG",
        hidden_layers_actor=(256, 128),  #
        hidden_layers_critic_body=(256,),  #
        hidden_layers_critic_head=(128,),  #
        func_critic_body="F.relu",  #
        func_critic_head="F.relu",  #
        func_actor_body="F.relu",  #
        lr_scheduler={'scheduler_type': "multistep",  # "step", "exp" or "decay", "multistep"
                      'gamma': 0.5,  # 0.99999,
                      'step_size': 1,
                      'milestones': [25*1000 * i for i in range(1, 6)],
                      'max_epochs': n_episodes},

        TAU=1e-3,  # for soft update of target parameters
        BUFFER_SIZE=int(1e6),  # replay buffer size
        BATCH_SIZE=128,  # minibatch size
        GAMMA=0.99,  # discount factor
        LR_ACTOR=1e-4,  # learning rate of the actor
        LR_CRITIC=3e-4,  # learning rate of the critic
        WEIGHT_DECAY=0.001,  # L2 weight decay
        UPDATE_EVERY=1,  # Number of actions before making a learning step
        action_noise="OU",  #
        action_noise_scale=1,
        weights_noise=None,  #
        state_normalizer=None, #"RunningMeanStd",  #
        warmup=0,  # Number of random actions to start with as a warm-up
        start_time=str(pd.Timestamp.utcnow()),
    )

    # ------------------------------------------------------------
    #  1. Initialization
    # ------------------------------------------------------------
    # 1. Start the Environment
    env = GymMonitor(config=config)

    # Actor model
    actor = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size, config["hidden_layers_actor"]),
                                func=F.tanh)
    actor_target = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size,
                                                                          config["hidden_layers_actor"]),
                                func=F.tanh)
    # Critic model
    critic = DeepNeuralNetHeadCritic(env.action_size,
                                     SimpleNeuralNetBody(env.state_size, config["hidden_layers_critic_body"],
                                                         func=eval(config["func_critic_body"])),
                                     hidden_layers_sizes=config["hidden_layers_critic_head"],
                                     func=eval(config["func_critic_head"]),
                                     end_func=None)
    critic_target = DeepNeuralNetHeadCritic(env.action_size,
                                     SimpleNeuralNetBody(env.state_size, config["hidden_layers_critic_body"],
                                                         func=eval(config["func_critic_body"])),
                                     hidden_layers_sizes=config["hidden_layers_critic_head"],
                                     func=eval(config["func_critic_head"]),
                                     end_func=None)

    # DDPG Agent
    agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                      model_actor=actor, model_critic=critic,
                      actor_target=actor_target, critic_target=critic_target,
                      action_space_low=-1, action_space_high=1,
                      config=config,
                      )

    env.eps_decay = config["eps_decay"]
    env.run(agent)

    # env.play(agent)


if __name__ == "__main__":

    for batch_size in [64]:
        for weight_decay in [0.000001, ]:
            for discount_factor in [0.9, ]:
                    main(discount_factor=discount_factor, weight_decay=weight_decay)
