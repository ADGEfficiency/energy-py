"""
Tests for the Flexibiliy environment
"""
import os
import pandas as pd
import numpy as np
from energy_py.envs import FlexEnv

data_path = os.path.join(os.getcwd(), 'data')

env = FlexEnv(data_path=data_path,
              flex_initial_size=1,
              flex_final_size=-1,
              flex_time=12,
              relax_time=6)

state = pd.read_csv('data/state.csv', index_col=0, parse_dates=True)

step = 0
rew = []
while step < 40:

    if step == 5:
        o, r, d, i = env.step(np.array([1]))
    else:
        o, r, d, i = env.step(np.array([0]))

    rew.append(r)
    step += 1

prices = state.values

initial_rew = np.sum(prices[5:5+12]) / 12
final_rew = -np.sum(prices[5+12:5+24]) / 12
print(initial_rew, final_rew, sum(rew))
assert (initial_rew + final_rew) == sum(rew)
