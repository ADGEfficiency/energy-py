""" tests for the energypy flex environment """

import numpy as np
import pandas as pd

import energypy


def check_energy_balance(info):
    inc_consumption = info['site_consumption'].sum() - info['site_demand'].sum()
    assert inc_consumption >= 0


def test_no_op():
    env = energypy.make_env('flex')

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


def test_increase_setpoint():
    env = energypy.make_env(
        'flex',
        capacity=4.0,
        supply_capacity=0.5,
        release_time=3,
        supply_power=0.05,
        episode_length=10,
        episode_sample='random'
    )

    obs = env.reset()
    env.seed(42)
    done = False
    step = 0

    while not done:
        act = np.array(0)

        if (step >= 2) and (step <= 4):
            act = np.array(1)
            print(step, act)

        next_obs, r, done, i = env.step(act)
        step += 1

    info = pd.DataFrame().from_dict(i)

    sub = info.loc[:, ['site_demand', 'site_consumption', 'setpoint']]

    np.testing.assert_array_equal(
        info.loc[:, 'site_consumption'].values[2:5],
        np.zeros(5-2)
    )

    np.testing.assert_array_almost_equal(
        info.loc[:, 'site_demand'].values[5:5+3] +
        info.loc[:, 'site_demand'].values[2:5],
        info.loc[:, 'site_consumption'].values[5:5+3]
    )


def test_decrease_setpoint():
    """ tests the precooling - but sets release time and capacity high """
    env = energypy.make_env(
        'flex',
        capacity=4.0,
        supply_capacity=100, # large to ignore the effect
        release_time=100,  #  large to ignore the effect
        supply_power=0.05,
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
            act = np.array(2)
            print(step, act)

        next_obs, r, done, i = env.step(act)
        step += 1

    info = pd.DataFrame().from_dict(i)

    sub = info.loc[:, ['site_demand', 'site_consumption', 'setpoint',
                       'stored_supply']]

    print(sub.head(15))

    np.testing.assert_array_equal(
        info.loc[:, 'site_consumption'].values[2:5],
        np.full(5-2, env.supply_power)
    )

    np.testing.assert_almost_equal(
        env.supply_power * 4 / 12 - np.sum(info.loc[:, 'site_demand'].values[2:6]) / 12,
        info.loc[:, 'stored_supply'][5]
    )
