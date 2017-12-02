import os

from energy_py.experiments import random_experiment, dqn_experiment, reinforce_experiment
from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv

    data_path = os.getcwd()
    random_outputs = random_experiment(env, 
                                       data_path=data_path,
                                       base_path='random/expt_1')

    dqn_outputs = dqn_experiment(env,
                                 data_path=data_path,
                                 base_path='dqn/expt_2')

    reinforce_outputs = reinforce_experiment(env,
                                             data_path=data_path,
                                             base_path='reinforce/expt_1')

