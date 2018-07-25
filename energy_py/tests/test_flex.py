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
    test_no_op()

    """ testing raising the setpoint - reducing cooling generation """
    rel_time = 4
    env = energy_py.make_env('flex', capacity=8.0, release_time=rel_time)

    obs = env.reset()
    done = False
    step = 0

    start = 3 

    while not done:
        act = np.array(0)

        if step >= start:
            act = np.array(1)

        next_obs, r, done, i = env.step(act)
        step += 1

    info = pd.DataFrame().from_dict(i)
    print(info.head(10))

    #  this will fail because agent can store demand at end of episode
    # check_energy_balance(info)

    cons = info.loc[:, 'site_consumption'].values[start+1:]
    dem = info.loc[:, 'site_demand'].values[start:-1]
    print(info.tail(20))

    for idx, (v1, v2) in enumerate(zip(cons, dem)):
        if (v1 != v2) and (idx < 10):
            print('step {}'.format(idx))
            print(v1, v2)
    print(cons[:10])
    print(dem[:10])

    print(cons[-10:])
    print(dem[-10:])

    import pdb; pdb.set_trace()

    np.testing.assert_almost_equal(cons, dem)




