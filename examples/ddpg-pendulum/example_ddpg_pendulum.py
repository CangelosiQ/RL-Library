"""
 Created by quentincangelosi at 28.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.monitors.openai_gym_monitor import GymMonitor

#TODO This are the parameters provided in ddpgagent.py by Udacity
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

env = GymMonitor('Pendulum-v0')
agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])

env.run(agent,
        n_episodes=20,
        length_episode=500,
        mode="train",
        reload_path=None,
        save_every=500,
        save_path="../../figures")
env.play(agent)