"""
 Created by quentincangelosi at 28.10.20
 From Global Advanced Analytics and Artificial Intelligence
"""
from rl_library.agents.ddpg_agent import DDPGAgent
from rl_library.monitors.openai_gym_monitor import GymMonitor


env = GymMonitor('BipedalWalker-v3')
agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])

env.run(agent,
        n_episodes=20,
        length_episode=500,
        mode="train",
        reload_path=None,
        save_every=500,
        save_path="../../figures")
env.play(agent)