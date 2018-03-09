import logging
import os

from energy_py import experiment
from energy_py.agents import DQN, DPG
from energy_py.envs import FlexEnv, BatteryEnv

if __name__ == '__main__':
    total_steps = 1e6
    agent_config = {'discount': 0.97,
                    'tau': 0.001,
                    'total_steps': total_steps,
                    'batch_size': 32,
                    'layers': (50, 50),
                    'learning_rate': 0.0001,
                    'initial_random': 0.0,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.4,
                    'memory_type': 'priority',
                    'process_observation': 'normalizer',
                    'process_target': 'standardizer'}

    env = BatteryEnv
    env_config = {'episode_length': 2016,
                  'episode_random': True}

    data_path = os.getcwd()+'/perfect_forecast/'
    results_path = os.getcwd()+'/results/priority_dqn_batt/'

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=agent_config,
                                  env=env,
                                  env_config=env_config,
                                  total_steps=total_steps,
                                  data_path=data_path,
                                  results_path=results_path)
