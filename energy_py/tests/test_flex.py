"""
Tests for the Flexibiliy environment
"""
import os
import pandas as pd
import numpy as np
from energy_py.envs import FlexEnv

data_path = os.path.join(os.getcwd(), 'data')

env = FlexEnv(data_path=data_path,
              flex_size=1,
              flex_time=6,
              relax_time=12,
              flex_effy=1.2)

state = pd.read_csv('data/state.csv', index_col=0, parse_dates=True)

prices = state.values

def test_down_up():
    o = env.reset()
    rews = []
    for step in range(30):

        if step == 5:
            o, r, d, i = env.step(np.array(1).reshape(1, 1))

        else:
            o, r, d, i = env.step(np.array(0).reshape(1, 1))

        rews.append(r)

    expected_rew = (np.sum(prices[5:5+6]) - np.sum(prices[5+6:5+6+6])*1.2) / 12
    assert np.isclose(sum(rews), expected_rew)


def test_up_down():
    o = env.reset()
    rews = []
    for step in range(30):

        if step == 6:
            o, r, d, i = env.step(np.array(2).reshape(1, 1))

        else:
            o, r, d, i = env.step(np.array(0).reshape(1, 1))

        rews.append(r)

    expected_rew = (-1.2*np.sum(prices[6:6+6]) + np.sum(prices[6+6:6+12])) / 12
    assert np.isclose(sum(rews), expected_rew)


if __name__ == '__main__':
    test_down_up()
    test_up_down()
