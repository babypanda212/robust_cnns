# src/training/methods/randomized_smoothing.py

import torch

class RandomizedSmoother:
    def __init__(self, sigma=0.25):
        self.sigma = sigma

    def add_noise(self, X):
        """Add Gaussian noise to inputs."""
        return X + torch.randn_like(X) * self.sigma
