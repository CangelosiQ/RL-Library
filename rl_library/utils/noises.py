import random
import numpy as np
import copy

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=42, scale=1., mu=0., theta=0.15, sigma=0.2, dt=1):
        """Initialize parameters and noise process."""
        self.seed = np.random.seed(seed)
        self.mu = mu * np.ones(size)
        self.scale = scale
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.reset()
        self.dt = dt

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.normal(size=self.size) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state * self.scale


# TODO Adapt
# import numpy as np
#
# Taken from https://github.com/openai/baselines
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, size, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.size = size

    def __call__(self):
        return np.random.normal(self.mu, self.sigma, size=self.size)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

