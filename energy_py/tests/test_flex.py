""" tests for the energy_py flex environment """

import numpy as np
import pandas as pd

import energy_py



def test_no_op():
    env = energy_py.make_env('flex')

    obs = env.reset()
    done = False
    step = 0

    while not done:
        act = np.array(0)
        next_obs, r, done, i = env.step(act)
        step += 1

    out = pd.DataFrame().from_dict(i)

    #  this little bit of computation should be in the flex env

    out['base_costs'] = out['site_demand'] * out['electricity_price']
    out['opt_costs'] = out['site_electricity_consumption'] * out['electricity_price']
    out['delta'] = out['base_costs'] - out['opt_costs']

    np.testing.assert_equal(out['delta'].sum(), 0)


def test_raise_sp():
    """ testing raising the setpoint - reducing cooling generation """
    env = energy_py.make_env('flex', capacity=2.0, release_time=4)

    obs = env.reset()
    done = False
    step = 0

    while not done:
        act = np.array(0)
        if step >= 3 and step < 7:
            act = np.array(1)

        next_obs, r, done, i = env.step(act)
        step += 1

    out = pd.DataFrame().from_dict(i)

    expt = out.iloc[:12, :]

    #  check that we charge and discharge equal amounts of energy
    np.testing.assert_equal(expt['stored'].sum(), expt['discharged'].sum())

    #  check the timing of the discharge
    #  making an assumption the capacity is big enough
    #  could fail if I change the example dataset
    #  maybe better to use a test dataset TODO
    np.testing.assert_array_equal(
        expt.loc[3:7, 'stored'], expt.loc[7:11, 'discharged']
    )

    np.testing.assert_array_equal(
        expt.loc[3, 'stored'], expt.loc[7, 'discharged']
    )


def test_lower_setpoint():
    """ testing reducing the setpoint - increasing cooling generation """

    #  long release time so that we release it all during
    #  the setpoint lowering
    env = energy_py.make_env(
        'flex', capacity=2.0, release_time=10
    )

    obs = env.reset()
    done = False
    step = 0

    while not done:
        act = np.array(0)
        if step >= 5 and step < 10:
            act = np.array(1)

        if step == 11:
            act = np.array(2)

        next_obs, r, done, i = env.step(act)
        step += 1

    out = pd.DataFrame().from_dict(i)

    expt = out.iloc[:11, :]

    #  check that we discharge everything that we stored
    np.testing.assert_equal(
        out.loc[:, 'stored'].sum(), out.loc[:, 'discharged'].sum()
    )

    #  check we only discharge when we reduce the setpoint
    np.testing.assert_equal(
        0, out.loc[:10, 'discharged'].sum()
    )

    np.testing.assert_equal(
        out.loc[:, 'stored'].sum(),
        out.loc[11:11, 'discharged'].sum()
    )


def test_release_when_full():
    #  long release time so that we release it all during
    #  the setpoint lowering
    capacity = 0.5
    env = energy_py.make_env(
        'flex', capacity=capacity, release_time=100
    )

    obs = env.reset()
    done = False
    step = 0

    while not done:
        #  always store
        act = np.array(1)

        next_obs, r, done, i = env.step(act)
        step += 1

    out = pd.DataFrame().from_dict(i)

    expt = out.iloc[:, :]

    #  calculate when we should have discharged

    cumulative_demand = []
    for idx, row in expt.iterrows():

        cumulative_demand.append(row.loc['site_demand'])
        if sum(cumulative_demand) >= capacity:
            np.testing.assert_almost_equal(
                row.loc['discharged'], sum(cumulative_demand)
            )
            cumulative_demand = []



if __name__ == '__main__':
    test_release_when_full()
