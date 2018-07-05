import os

import numpy as np

import energy_py

default_config = {
    'env_id': 'battery',
     'dataset': 'example',
     'initial_charge': 0,
     'round_trip_eff': 0.9
}

#  this code is just to get the charge index
#  so I don't need to do it in each test function
env = energy_py.make_env(**default_config)
charge_index = env.observation_space.info.index('C_charge_level [MWh]')


def test_charge():
    config = default_config 

    env = energy_py.make_env(**config)

    output = env.step(action=np.array([1.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]

    expected_charge = 0 + 0.9 * 1.0 / 12

    assert charge == expected_charge


def test_discharge():
    config = default_config
    config['initial_charge'] = 1.0
    config['capacity'] = 4.0

    env = energy_py.make_env(**config)

    output = env.step(action=np.array([-1.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 4.0 - 1.0 / 12

    assert charge == expected_charge


def test_no_op():
    config = default_config
    config['initial_charge'] = 0.5 
    config['capacity'] = 4.0

    env = energy_py.make_env(**config)

    output = env.step(action=np.array([0.0]).reshape(1, 1))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 2.0
    assert charge == expected_charge
