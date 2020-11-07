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
log_fn = f"../../logs/logs_{__file__.split('/')[-1].replace('.py', '')}_{pd.Timestamp.utcnow().value}.log"
logging.basicConfig(filename=log_fn,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------------
#  Internal Dependencies
# ---------------------------------------------------------------------------------------------------
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.openai_gym_monitor import GymMonitor


# ---------------------------------------------------------------------------------------------------
#  Inputs
# ---------------------------------------------------------------------------------------------------
config = dict(
    # Environment parameters
    env_name='BipedalWalker-v3',
    n_episodes=500,
    length_episode=1500,
    save_every=50,
    save_path="../../figures",
    mode="train",  # "train" or "test"
    evaluate_every=10,                   # Number of training episodes before 1 evaluation episode
    eps_decay=0.99,                      # Epsilon decay rate
    render=True,

    # Agent Parameters
    agent="DDPG",
    hidden_layers_actor=(40, 30,),          #
    hidden_layers_critic_body=(40, 30,),    #
    hidden_layers_critic_head=(30, 15),    #
    func_critic_body="F.relu",          #
    func_critic_head="F.relu",          #
    func_actor_body="F.relu",           #
    TAU=1e-3,                           # for soft update of target parameters
    BUFFER_SIZE=int(1e6),               # replay buffer size
    BATCH_SIZE=64,                      # minibatch size
    GAMMA=0.99,                          # discount factor
    LR_ACTOR=1e-4,                      # learning rate of the actor
    LR_CRITIC=1e-3,                     # learning rate of the critic
    WEIGHT_DECAY=0.01,                  # L2 weight decay
    UPDATE_EVERY=1,                     # Number of actions before making a learning step
    action_noise="OU",                  #
    weights_noise=None,                 #
    batch_normalization=None,           #
    warmup=0,                           # Number of random actions to start with as a warm-up
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
critic = DeepNeuralNetHeadCritic(env.action_size, SimpleNeuralNetBody(env.state_size, config["hidden_layers_critic_body"],
                                                                  func=eval(config["func_critic_body"])),
                                 hidden_layers_sizes=config["hidden_layers_critic_head"],
                                 func=eval(config["func_critic_head"]),
                                 end_func=None)

# DDPG Agent
agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                  model_actor=actor, model_critic=critic,
                  action_space_low=[-1, ] * env.action_size, action_space_high=[1, ] * env.action_size,
                  config={"warmup": config["warmup"], "action_noise": config["action_noise"]},
                  hyper_parameters=config)

env.eps_decay = config["eps_decay"]
env.run(agent)

env.play(agent)

with open(f"./{config['save_path']}/config.json", "w") as f:
    json.dump(config, f)
