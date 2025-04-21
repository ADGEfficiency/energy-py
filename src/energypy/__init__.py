"""Reinforcement learning experiments with energy environments with energypy."""

import gymnasium as gym

from energypy.battery import Battery
from energypy.experiment import run_experiment

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)
__all__ = [
    "Battery",
    "run_experiment",
]
