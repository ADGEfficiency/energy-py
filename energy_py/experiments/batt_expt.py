import logging
import os

from energy_py import experiment
from energy_py.agents import DQN
from energy_py.envs import CartPoleEnv, FlexEnv, BatteryEnv

if __name__ == '__main__':

    agent = DQN
    agent_config = {'discount': 0.97,
                    'tau': 0.001,
                    'total_steps': 500000,
                    'batch_size': 32,
                    'layers': (50, 50),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.4,
                    'process_observation': False,
                    'process_target': False}

    env = BatteryEnv
    env_config = {'episode_length': 2016,
                  'episode_random': True}

    total_steps = 1e6

    data_path = './perfect_forecast',
    results_path = '.results/perfect_battery_dqn'

    info = experiment(agent, agent_config, env, env_config,
                      total_steps, data_path, results_path)
