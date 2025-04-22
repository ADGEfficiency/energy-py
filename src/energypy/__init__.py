"""Reinforcement learning experiments with energy environments with energypy."""

import gymnasium as gym

from energypy.battery import Battery
from energypy.experiment import ExperimentConfig, run_experiment, run_experiments

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)

__all__ = [
    "Battery",
    "ExperimentConfig",
    "run_experiment",
    "run_experiments",
]
