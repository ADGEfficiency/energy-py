from collections import OrderedDict, defaultdict

import numpy as np
import pytest

from energypy.registry import make
"""
- test cursor position (through info dict)
- test the shapes of obs & action spaces

"""


test_cases = (
    #  full charge for three steps
    (
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [1.0, 1.0, 1.0],
        [2.0/2, 4.0/2, 6.0/2]
    ),
    #  full, half then full charge for three steps
    (
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [1.0, 0.5, 1.0],
        [2.0/2, 3.0/2, 5.0/2]
    ),
    (
    #   discharge, charge, discharge
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [-1.0, 1.0, -1.0],
        [0.0, 2.0/2, 0.0]
    )
)


@pytest.mark.parametrize('cfg, actions, expected_charges', test_cases)
def test_one_battery_charging(cfg, actions, expected_charges):
    env = make('battery', **cfg, n_batteries=1)
    env.reset()

    results = defaultdict(list)
    for action in actions:
        action = np.array(action).reshape(1, 1)
        next_obs, reward, done, info = env.step(action)
        results['charge'].append(info['charge'])

    assert done
    charges = np.squeeze(np.array(results['charge']))
    np.testing.assert_array_almost_equal(charges, expected_charges)


def test_battery_init():
    env = make(
        'battery',
        dataset={'name': 'random-dataset', 'n_features': 16}
    )
    #  can check shapes of dataset, action space etc


def test_many_battery_step():
    cfgs = defaultdict(list)

    actions, charges = [], []
    for test_case in test_cases:

        #  the config dict
        for k, v in test_case[0].items():
            cfgs[k].append(v)

        actions.append(test_case[1])
        charges.append(test_case[2])

    cfgs['episode_length'] = 3
    #  actions = (3, 3)
    #  needs to be timestep first!
    actions = np.array(actions).T
    expected_charges = np.array(charges).T

    env = make(
        'battery',
        n_batteries=len(test_cases),
        **cfgs,
        dataset={'name': 'random-dataset', 'n_features': 10}
    )

    #  test 1
    np.testing.assert_array_equal(cfgs['power'], env.power[0, 0])
    assert env.power.shape == (len(test_cases), 1)

    obs = env.reset()
    results = defaultdict(list)
    for action in actions:
        action = np.array(action).reshape(len(test_cases), 1)
        next_obs, reward, done, info = env.step(action)
        print(env.charge, 'charge')
        results['charge'].append(info['charge'])
        #  1 for the charge variable added onto our 10 features
        assert next_obs.shape == (len(test_cases), 10+1)

    assert done.all()
    np.testing.assert_array_almost_equal(
        np.squeeze(results['charge']),
        np.squeeze(expected_charges)
    )
