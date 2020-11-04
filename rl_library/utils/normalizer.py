"""
 Created by quentincangelosi at 03.11.20
 From Global Advanced Analytics and Artificial Intelligence
"""
import numpy as np


# Inspired from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/normalizer.py
class MeanStdNormalizer:
    def __init__(self, clip=10.0, epsilon=1e-8, use_running_values=False):
        self.use_running_values = use_running_values
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon
        self.sum = epsilon
        self.mean = epsilon
        self.var = epsilon

    def __call__(self, x, read_only = False):
        x = np.asarray(x)
        if self.use_running_values:
            raise NotImplementedError#self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        elif not read_only:
            self.mean = np.mean(x, 0)
            self.var = np.var(x, 0)
        # print(f"x={x}, mean={self.mean}, std={self.var}")
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon),
                       -self.clip, self.clip)