"""The true price."""

import numpy as np


class RandomWalk:
    """S_{t+1} = S_t + sigma * eps."""

    def __init__(self, S0=100.0, sigma=0.1):
        self.S0 = S0
        self.sigma = sigma
        self.S = S0

    def reset(self):
        self.S = self.S0
        return self.S

    def step(self, rng):
        self.S += self.sigma * rng.standard_normal()
        return self.S
