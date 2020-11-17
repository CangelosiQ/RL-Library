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
from baselines.ddpg.ddpg import learn


def main():
    learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None,  # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64,  # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs)

    # env.play(agent)


if __name__ == "__main__":
    main()
