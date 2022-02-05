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
    n_samples = 4

    data = []
    for rw in range(n_samples):
        data.append({
            'features': np.random.random((episode_len, sequence_length, n_features)),
            'mask': np.random.random((episode_len, sequence_length, sequence_length)),
            'prices': np.random.random((episode_len, 1))
        })

    ds = NEMDatasetAttention(
        n_batteries=2,
        train_episodes=data,
        test_episodes=data
    )

    #  not fine here
    ds.setup_test()
    #  not fine here
    obs = ds.reset('test')

    #  check the first feature is correct
    f0 = ds.sample_observation(0)['features'][0]

    f0check = data[0]['features'][0]
    assert f0.shape == (sequence_length, n_features)
    np.testing.assert_array_equal(f0, f0check)


def test_attention_dataset_load_arrays():
    #  make dummy dataset
    episode_len = 8
    n_batteries = 3
    sequence_length = 4
    n_features = 5

    dss = [
        {
            'features': np.random.random((episode_len, n_batteries, sequence_length, n_features)),
            'mask': np.random.randint(0, 1, episode_len * n_batteries * sequence_length * sequence_length).reshape(episode_len, n_batteries, sequence_length, sequence_length),
            'prices': np.random.random((episode_len, n_batteries, sequence_length, n_features))
        },
        {
            'features': np.random.random((episode_len, n_batteries, sequence_length, n_features)),
            'mask': np.random.randint(0, 1, episode_len * n_batteries * sequence_length * sequence_length).reshape(episode_len, n_batteries, sequence_length, sequence_length),
            'prices': np.random.random((episode_len, n_batteries, sequence_length, n_features))
        },
        {
            'features': np.random.random((episode_len, n_batteries, sequence_length, n_features)),
            'mask': np.random.randint(0, 1, episode_len * n_batteries * sequence_length * sequence_length).reshape(episode_len, n_batteries, sequence_length, sequence_length),
            'prices': np.random.random((episode_len, n_batteries, sequence_length, n_features))
        }
    ]

    from pathlib import Path
    path = Path('./temp/train')
    for ds, date in zip(dss, ['2020-01-01', '2021-01-01', '2022-01-01']):
        for name, data in ds.items():
            (path / name).mkdir(exist_ok=True, parents=True)
            np.save(path / name / date, data)

    #  load
    ds = NEMDatasetAttention(n_batteries, './temp/train')
