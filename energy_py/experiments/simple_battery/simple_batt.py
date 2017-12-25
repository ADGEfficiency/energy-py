import os

from energy_py.agents import RandomAgent, NaiveBatteryAgent, DQN, DPG

from energy_py.experiments import no_learning_experiment
from energy_py.experiments import dqn_experiment
from energy_py.experiments import dpg_experiment

from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv
    data_path = os.getcwd()

    for exp in range(1):
        # naive_outputs = no_learning_experiment(NaiveBatteryAgent,
        #                                        env,
        #                                        data_path,
        #                                        'naive')

        # dqn_outputs = dqn_experiment(DQN,
        #                              env,
        #                              data_path,
        #                              base_path='dqn',
        #                              opt_parser_args={'name': '--bs',
        #                                               'type': int,
        #                                               'default': 64,
        #                                               'help': 'batch size for experience replay'})

        dpg_out = dpg_experiment(DPG,
                                 env,
                                 data_path,
                                 base_path='dpg')
