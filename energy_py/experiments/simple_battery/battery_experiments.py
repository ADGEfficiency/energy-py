import os

from energy_py.agents import RandomAgent, NaiveBatteryAgent
from energy_py.experiments import no_learning_experiment
from energy_py.experiments import dqn_experiment
from energy_py.experiments import reinforce_experiment
from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv
    data_path = os.getcwd()

    for exp in range(1):
        naive_outputs = no_learning_experiment(NaiveBatteryAgent,
                                               env,
                                               data_path=data_path)

        dqn = dqn_experiment(env, data_path)
