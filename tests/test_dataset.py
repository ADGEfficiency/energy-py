from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import pytest

from energypy.registry import make
from energypy.datasets import make_perfect_forecast, NEMDatasetAttention


def test_make_random_dataset_one_battery():
    env = make('battery', n_batteries=1, dataset={'name': 'random-dataset', 'n': 10000, 'n_features': 3})

    dataset = env.dataset.dataset

    assert dataset['prices'].shape[0] == 10000
    assert dataset['features'].shape[0] == 10000

    assert len(dataset['prices'].shape) == 3
    assert dataset['features'].shape[1] == 1
    assert dataset['features'].shape[2] == 3


def test_make_random_dataset_many_battery():
    env = make(
        'battery',
        n_batteries=4,
        dataset={
            'name': 'random-dataset',
            'n': 1000,
            'n_features': 6,
        }
    )

    data = env.dataset.dataset
    print(data['prices'].shape, data['features'].shape)
    assert data['prices'].shape[0] == 1000

    #  (timestep, battery, features)
    assert data['features'].shape[0] == 1000
    assert data['features'].shape[1] == 4
    assert data['features'].shape[2] == 6


def test_make_perfect_forecast():
    prices = np.array([10, 50, 90, 70, 40])
    horizon = 3
    forecast = make_perfect_forecast(prices, horizon)

    expected = np.array([
        [10, 50, 90],
        [50, 90, 70],
        [90, 70, 40],
    ])
    np.testing.assert_array_equal(expected, forecast)


def test_attention_dataset():
    episode_len = 32
    sequence_length = 4
    n_features = 5

    #  three episodes
    features = [
        np.random.random((episode_len, sequence_length, n_features)),
        np.random.random((episode_len, sequence_length, n_features)),
        np.random.random((episode_len, sequence_length, n_features)),
    ]
    masks = [
        np.random.random((episode_len, sequence_length, sequence_length)),
        np.random.random((episode_len, sequence_length, sequence_length)),
        np.random.random((episode_len, sequence_length, sequence_length)),
    ]

    prices = [np.random.random((episode_len, 1)) for _ in range(3)]

    ds = NEMDatasetAttention(
        n_batteries=3,
        train_episodes={'features': features, 'mask': masks, 'prices': prices}
    )

    #  bit of a smell
    ds.setup_test()
    obs = ds.reset('test')

    #  check the first feature is correct
    features0 = ds.sample_observation(0)
    assert features0 == features[0]
    assert features0.shape == (1, n_batteries, sequence_length, n_features)

    #  can also check ds.episode dict ?


test_attention_dataset()
