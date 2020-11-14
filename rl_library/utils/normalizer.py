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

    def __call__(self, x, read_only=False):
        x = np.asarray(x)
        if self.use_running_values:
            raise NotImplementedError  # self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        elif not read_only:
            self.mean = np.mean(x, 0)
            self.var = np.var(x, 0)
        # print(f"x={x}, mean={self.mean}, std={self.var}")
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon),
                       -self.clip, self.clip)


# Based on https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-8, shape=(), clip=10):
        print(f"shape={shape}")
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def __call__(self, x, read_only=False):
        x = np.asarray(x)
        if not read_only:
            self.update(x)
        # print(f"x={x}, mean={self.mean}, std={self.var}")
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon),
                       -self.clip, self.clip)

    def __str__(self):
        return f"RunningMeanStd(mean={self.mean}, var={self.var}, count={self.count})"

# Based on https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
