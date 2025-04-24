import energypy
import gymnasium as gym
import numpy as np


def test_readme() -> None:
    # env = energypy.Battery()
    # results = energypy.train(env, "PPO", name="battery")
    pass


def test_make_env() -> None:
    """Test that make_env creates and returns a properly configured environment."""
    # Create test electricity prices
    electricity_prices = [50.0] * 100
    
    # Test with only electricity_prices
    env = energypy.make_env(electricity_prices=electricity_prices)
    assert isinstance(env, gym.wrappers.NormalizeReward)
    
    # Test with features
    features = {"feature1": np.array([1.0, 2.0]), "feature2": np.array([3.0, 4.0])}
    env_with_features = energypy.make_env(electricity_prices=electricity_prices, features=features)
    assert isinstance(env_with_features, gym.wrappers.NormalizeReward)
