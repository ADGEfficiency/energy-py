"""Reinforcement learning experiments with energy environments with energypy."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from energypy.battery import Battery
from energypy.experiment import ExperimentConfig, run_experiment, run_experiments

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)


def make_env(electricity_prices, features=None, freq_mins=60):
    """
    Create a battery environment with electricity prices and optional features.

    Args:
        electricity_prices: A sequence of electricity prices
        features: Optional features array with same length as prices.
                 If None, uses electricity_prices reshaped as features.
        freq_mins: The number of minutes each step represents (default: 60 for hourly)

    Returns:
        A normalized battery environment
    """
    # If features is None, use the electricity prices as features
    if features is None:
        # Reshape prices to make it a 2D array with shape (n, 1)
        prices_array = np.array(electricity_prices)
        features = prices_array.reshape(-1, 1)

    env = gym.make(
        "energypy/battery",
        electricity_prices=electricity_prices,
        features=features,
        freq_mins=freq_mins,
    )
    env = gym.wrappers.NormalizeReward(env)
    env = Monitor(env, filename="./data/data.log")

    def env_fn():
        return cast(Env[Any, Any], env)

    vec_env = DummyVecEnv([env_fn])
    return vec_env


__all__ = [
    "Battery",
    "ExperimentConfig",
    "run_experiment",
    "run_experiments",
    "make_env",
]
