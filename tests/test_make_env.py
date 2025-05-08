import gymnasium as gym
import numpy as np

import energypy


def test_make_env() -> None:
    """Test that make_env creates and returns a properly configured environment."""
    # Create test electricity prices
    electricity_prices = [50.0] * 100

    # Test with only electricity_prices
    env = energypy.make_env(electricity_prices=electricity_prices)
    assert isinstance(env, gym.wrappers.NormalizeReward)

    # Test with features
    features = np.random.uniform(-100.0, 100, (100, 4))
    env_with_features = energypy.make_env(
        electricity_prices=electricity_prices, features=features
    )
    assert isinstance(env_with_features, gym.wrappers.NormalizeReward)
