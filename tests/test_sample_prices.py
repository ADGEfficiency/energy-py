import numpy as np


def sample_prices(prices, n_lags, episode_length):
    start = np.random.randint(n_lags, len(prices) - n_lags - episode_length)
    end = start + n_lags + episode_length


def test_sample_prices() -> None:
    prices = np.random.uniform(0, 1000, 1000)
    n_lags = 20
    episide_length = 20

    sample = sample_prices(prices, n_lags=n_lags, episode_length=episide_length)
    assert sample.shape == (n_lags + episide_length, 1)
