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
        length_episode=700,
        save_every=100,
        save_path=f"./DDPG_Bipedal_{pd.Timestamp.utcnow().value}",
        mode="train",  # "train" or "test"
        evaluate_every=50,  # Number of training episodes before 1 evaluation episode
        eps_decay=0.995,  # Epsilon decay rate
        render=False,

        # Agent Parameters
        agent="DDPG",
        hidden_layers_actor=(100, 30),  #
        hidden_layers_critic_body=(100,),  #
        hidden_layers_critic_head=(30,),  #
        func_critic_body="F.relu",  #
        func_critic_head="F.relu",  #
        func_actor_body="F.relu",  #
        lr_scheduler={'scheduler_type': "exp",  # "step", "exp" or "decay"
                      'gamma': 0.99995,
                      'step_size': 1,
                      'max_epochs': n_episodes},
        # Hyper Parameters Agent Models
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
    env = GymMonitor(config=config)

    # Actor model
    actor = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size, config["hidden_layers_actor"]),
                                func=torch.tanh)
    # Critic model
    critic = DeepNeuralNetHeadCritic(env.action_size,
                                     SimpleNeuralNetBody(env.state_size, config["hidden_layers_critic_body"],
                                                         func=eval(config["func_critic_body"])),
                                     hidden_layers_sizes=config["hidden_layers_critic_head"],
                                     func=eval(config["func_critic_head"]),
                                     end_func=None)

    # DDPG Agent
    agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                      model_actor=actor, model_critic=critic,
                      action_space_low=[-1, ] * env.action_size, action_space_high=[1, ] * env.action_size,
                      config=config,
                      )

    env.eps_decay = config["eps_decay"]
    env.run(agent)

    # env.play(agent)

    with open(f"./{config['save_path']}/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":

    for batch_size in [64, 128]:
        for weight_decay in [0, 0.0001]:
            for discount_factor in [0.6, 0.8]:
                    main(discount_factor=discount_factor, weight_decay=weight_decay)
