""" tests for the energy_py flex environment """

import numpy as np
import pandas as pd

import energy_py


def check_energy_balance(info):
    inc_consumption = info['site_consumption'].sum() - info['site_demand'].sum()
    assert inc_consumption >= 0


def test_no_op():
    env = energy_py.make_env('flex')

    obs = env.reset()
    done = False
    step = 0

    while not done:
        act = np.array(0)
        next_obs, r, done, i = env.step(act)
        step += 1

    info = pd.DataFrame().from_dict(i)

    check_energy_balance(info)

    np.testing.assert_equal(info['reward'].sum(), 0)


if __name__ == '__main__':
    env = energy_py.make_env(
        'flex',
        capacity=4.0,
        precool_capacity=0.5,
        release_time=3,
        precool_power=0.05,
        episode_length=10,
        episode_sample='random'
    )

    obs = env.reset()
    env.seed(42)
    done = False
    step = 0

    while not done:
        act = np.array(0)

        if (step >= 2) and (step <= 5):
            act = np.array(1)
            print(step, act)

        next_obs, r, done, i = env.step(act)
        step += 1

    info = pd.DataFrame().from_dict(i)

    sub = info.loc[:, ['site_demand', 'site_consumption', 'setpoint']]
    # print(info.describe())
    print(sub.head(15))
