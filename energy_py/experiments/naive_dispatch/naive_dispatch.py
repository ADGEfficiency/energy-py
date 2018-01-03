import os

from energy_py.agents import DispatchAgent
from energy_py.envs import FlexEnv
from energy_py.experiments import no_learning_experiment

if __name__ == '__main__':

    data_path = os.getcwd()

    env = FlexEnv

    dispatch = no_learning_experiment(DispatchAgent,
                                      env,
                                      data_path,
                                      'dispatch')


