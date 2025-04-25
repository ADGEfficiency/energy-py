"""Reinforcement learning experiments with energy environments with energypy."""

import gymnasium as gym

from energypy.battery import Battery
from energypy.experiment import ExperimentConfig, run_experiment, run_experiments

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)


def make_env(electricity_prices, features=None):
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
