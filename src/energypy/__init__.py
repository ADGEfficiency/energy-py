"""Reinforcement learning experiments with energy environments with energypy."""

import gymnasium as gym
import numpy as np

from energypy.battery import Battery
from energypy.experiment import ExperimentConfig, run_experiment, run_experiments

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)


def make_env(electricity_prices, features=None):
    """
    Create a battery environment with electricity prices and optional features.
    
    Args:
        electricity_prices: A sequence of electricity prices
        features: Optional features array with same length as prices. 
                 If None, uses electricity_prices reshaped as features.
                 
    Returns:
        A normalized battery environment
    """
    # If features is None, use the electricity prices as features
    if features is None:
        # Reshape prices to make it a 2D array with shape (n, 1)
        prices_array = np.array(electricity_prices)
        features = prices_array.reshape(-1, 1)
    
    env = gym.make(
        "energypy/battery", electricity_prices=electricity_prices, features=features
    )
    env = gym.wrappers.NormalizeReward(env)
    return env


__all__ = [
    "Battery",
    "ExperimentConfig",
    "run_experiment",
    "run_experiments",
    "make_env",
]
