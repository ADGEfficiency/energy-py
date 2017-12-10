import os

from energy_py.experiments import random_experiment
from energy_py.experiments import dqn_experiment
from energy_py.experiments import reinforce_experiment
from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv
    data_path = os.getcwd()

    # for exp in range(1):
    #     random_outputs = random_experiment(env,
    #                                        data_path=data_path,
    #                                        base_path='random/expt_{}'.format(exp))
    for exp in range(1,2):
        dqn_outputs = dqn_experiment(env,
                                     data_path=data_path,
                                     base_path='dqn/expt_{}'.format(exp))

    # reinforce_outputs = reinforce_experiment(env,
    #                                          data_path=data_path,
    #                                          base_path='reinforce/expt_0')
