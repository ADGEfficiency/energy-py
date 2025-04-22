"""Reinforcement learning experiments with energy environments with energypy."""

import gymnasium as gym

# Fix import cycle by using relative imports
from .battery import Battery

gym.register(
    id="energypy/battery",
    entry_point="energypy:Battery",
)

# Import after gym registration to prevent circular dependency
from .experiment import run_experiment, run_experiments

__all__ = [
    "Battery",
    "run_experiment",
    "run_experiments",
]
