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
logger = logging.getLogger()

# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Setup log file

# file_handler = logging.FileHandler()
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.DEBUG)
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# logger.addHandler(file_handler)
# logger.addHandler(stream_handler)

from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.openai_gym_monitor import GymMonitor


seed = torch.manual_seed(42)

env = GymMonitor(env_name='BipedalWalker-v3')

actor = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size, (50, 30,), func=F.leaky_relu),
                            func=F.tanh)

critic = DeepNeuralNetHeadCritic(env.action_size, SimpleNeuralNetBody(env.state_size, (50, 30),
                                                                      func=F.leaky_relu),
                                 hidden_layers_sizes=(30,),
                                 end_func=None,
                                 func=F.leaky_relu)

agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                  model_actor=actor, model_critic=critic, action_space_high=env.action_space.high,
                  action_space_low=env.action_space.low)

env.run(agent,
        n_episodes=2000,
        length_episode=1000,
        mode="train",
        reload_path=None,
        save_every=2000,
        render=True,
        save_path="../../figures")
env.play(agent)
