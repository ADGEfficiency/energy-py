import os

from energy_py.agents import NaiveBatteryAgent, RandomAgent

from energy_py.experiments import no_learning_experiment 
from energy_py.experiments import dqn_experiment
from energy_py.experiments import reinforce_experiment

from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv
    data_path = os.getcwd()

    naive = no_learning_experiment(NaiveBatteryAgent,
                                   env,
                                   data_path,
                                   'naive_agent')

    random = no_learning_experiment(RandomAgent,
                                    env,
                                    data_path,
                                    'random_agent')

