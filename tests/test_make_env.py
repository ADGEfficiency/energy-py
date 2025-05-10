import numpy as np

import energypy


def test_make_env() -> None:
    """Test that make_env creates and returns a properly configured environment."""
    electricity_prices = [50.0] * 100
    energypy.make_env(electricity_prices=electricity_prices)

    features = np.random.uniform(-100.0, 100, (100, 4))
    energypy.make_env(electricity_prices=electricity_prices, features=features)
