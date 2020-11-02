"""
 Created by quentincangelosi at 28.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import torch.nn.functional as F
import torch
from gym.spaces import Box

from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.agents.models.bodies import SimpleNeuralNetBody, SimpleNeuralNetBodyCritic
from rl_library.agents.models.heads import SimpleNeuralNetHead, DeepNeuralNetHeadCritic
from rl_library.monitors.openai_gym_monitor import GymMonitor

seed = torch.manual_seed(42)

env = GymMonitor('BipedalWalker-v3')

actor = SimpleNeuralNetHead(env.action_size, SimpleNeuralNetBody(env.state_size, (20, 16, 8)),
                            func=F.tanh)

critic = DeepNeuralNetHeadCritic(env.action_size, SimpleNeuralNetBody(env.state_size, (24, 18,),
                                                                      func=F.leaky_relu),
                                 hidden_layers_sizes=(12, 6,),
                                 end_func=None)

agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size,
                  model_actor=actor, model_critic=critic)

env.run(agent,
        n_episodes=2000,
        length_episode=1000,
        mode="train",
        reload_path=None,
        save_every=200,
        save_path="../../figures")
env.play(agent)
