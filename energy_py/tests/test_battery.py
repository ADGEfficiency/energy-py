import os

import numpy as np

import energy_py

config = {'env_id': 'Battery',
          'dataset': 'example',
          'initial_charge': 0,
          'round_trip_eff': 0.9}

#  this code is just to get the charge index
#  so I don't need to do it in each test function
env = energy_py.make_env(**config)
charge_index = env.observation_info.index('C_charge_level_[MWh]')


def test_charge():
    config = {'env_id': 'Battery',
              'dataset': 'example',
              'initial_charge': 0.0,
              'round_trip_eff': 0.9}

    env = energy_py.make_env(**config)

    output = env.step(action=np.array([1.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]

    expected_charge = 0 + 0.9 * 1.0 / 12

    assert charge == expected_charge


def test_discharge():
    config = {'env_id': 'Battery',
              'dataset': 'example',
              'capacity': 4.0,
              'initial_charge': 1.0,
              'round_trip_eff': 0.9}
    env = energy_py.make_env(**config)

    output = env.step(action=np.array([-1.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 4.0 - 1.0 / 12

    assert charge == expected_charge


def test_no_op():

    config = {'env_id': 'Battery',
              'dataset': 'example',
              'capacity': 4.0,
              'initial_charge': 0.5,
              'round_trip_eff': 0.9}
    env = energy_py.make_env(**config)

    output = env.step(action=np.array([0.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 2.0
    assert charge == expected_charge

