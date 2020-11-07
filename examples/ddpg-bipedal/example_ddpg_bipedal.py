"""
 Created by quentincangelosi at 28.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import torch.nn.functional as F
import torch
from gym.spaces import Box
import logging
import pandas as pd

log_fn = f"../../logs/logs_{__file__.split('/')[-1].replace('.py', '')}_{pd.Timestamp.utcnow().value}.log"
logging.basicConfig(filename=log_fn,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger("rllib")

from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.openai_gym_monitor import GymMonitor

seed = torch.manual_seed(42)

env = GymMonitor(env_name='BipedalWalker-v3')

actor = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size, (60, 40, 20), func=F.relu),
                            func=F.tanh)

critic = DeepNeuralNetHeadCritic(env.action_size, SimpleNeuralNetBody(env.state_size, (60, ),
                                                                      func=F.relu),
                                 hidden_layers_sizes=(40, 20),
                                 end_func=None,
                                 func=F.relu)

agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                  model_actor=actor, model_critic=critic, action_space_high=env.action_space.high,
                  action_space_low=env.action_space.low, config=dict(warmup=1e4,
                                                                     ),
                  hyper_parameters=dict(TAU=1e-3,  # for soft update of target parameters
                                        BUFFER_SIZE=int(1e5),  # replay buffer size
                                        BATCH_SIZE=128,  # minibatch size
                                        GAMMA=0.99,  # discount factor
                                        LR_ACTOR=1e-3,  # learning rate of the actor
                                        LR_CRITIC=1e-3,  # learning rate of the critic
                                        WEIGHT_DECAY=0.001,  # L2 weight decay
                                        UPDATE_EVERY=1,
                                        ))

env.run(agent,
        n_episodes=2000,
        length_episode=1000,
        mode="train",
        reload_path=None,
        save_every=100,
        render=False,
        save_path="../../figures")
env.play(agent)
