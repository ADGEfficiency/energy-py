import os
from energy_py.envs import BatteryEnv
import numpy as np

data_path = os.path.join(os.getcwd(), 'data')

env = BatteryEnv(data_path=data_path,
                 initial_charge=0,
                 round_trip_eff=0.9)

charge_index = env.observation_info.index('C_charge_level_[MWh]')


def test_charge():
    env = BatteryEnv(data_path=data_path,
                     initial_charge=0,
                     round_trip_eff=0.9)
    output = env.step(action=np.array([1.0, 0.0]).reshape(1,2))
    observation = output[0]
    charge = observation[0][charge_index]

    expected_charge = 0 + 0.9 * 1.0 / 12

    assert charge == expected_charge


def test_discharge():
    env = BatteryEnv(data_path=data_path,
                     capacity=4,
                     initial_charge=1.0,
                     round_trip_eff=0.9)

    output = env.step(action=np.array([0.0, 1.0]).reshape(1, 2))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 4.0 - 1.0 / 12

    assert charge == expected_charge


def test_no_op():
    env = BatteryEnv(data_path=data_path,
                     capacity=4,
                     initial_charge=0.5,
                     round_trip_eff=0.9)

    output = env.step(action=np.array([0.0, 0.0]).reshape(1, 2))
    observation = output[0]
    charge = observation[0][charge_index]
    expected_charge = 2.0
    assert charge == expected_charge

    output = env.step(action=np.array([2.0, 2.0]).reshape(1, 2))
    observation = output[0]
    charge = observation[0][charge_index]
    assert charge == expected_charge
