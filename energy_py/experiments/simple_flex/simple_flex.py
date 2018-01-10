import os

from energy_py.agents import DQN 
from energy_py.envs import FlexEnv
from energy_py.experiments import dqn_experiment


if __name__ == '__main__':

    data_path = os.getcwd()

    env = FlexEnv

    dqn = dqn_experiment(DQN,
                         env,
                         data_path,
                         opt_parser_args={'name': '--bs',
                                          'type': int,
                                          'default': 64,
                                          'help': 'batch size for experience replay'}
