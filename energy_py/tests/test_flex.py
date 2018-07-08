""" tests for the energy_py flex environment """

import numpy as np
import pandas as pd

import energy_py


if __name__ == '__main__':

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


    def test_lower_sp():
        #  test 2 - test action = 1
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

        #  this little bit of computation should be in the flex env

        out['base_costs'] = out['site_demand'] * out['electricity_price']
        out['opt_costs'] = out['site_electricity_consumption'] * out['electricity_price']
        out['delta'] = out['base_costs'] - out['opt_costs']

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
