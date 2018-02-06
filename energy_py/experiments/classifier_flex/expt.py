import os

from energy_py.agents import ClassifierAgent
from energy_py.envs import FlexEnv
from energy_py import experient

if __name__ == '__main__':

    data_path = os.getcwd()

    env = FlexEnv

    random = no_learning_experiment(ClassifierAgent,
                                    env,
                                    data_path,
                                    base_path='class')
