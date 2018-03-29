import os

from energy_py import experiment
from energy_py.agents import DQN, DPG
from energy_py.envs import FlexEnv, BatteryEnv

if __name__ == '__main__':
    total_steps = 1e6
    agent_config = {'discount': 0.99,
                    'tau': 0.001,
                    'total_steps': total_steps,
                    'batch_size': 32,
                    'layers': (25, 25, 25),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.4,
                    'memory_fraction': 0.15,
                    'memory_type': 'deque',
                    'double_q': False,
                    'process_observation': 'standardizer',
                    'process_target': 'normalizer'}

    env = BatteryEnv
    env_config = {'episode_length': 2016,
                  'initial_charge': 'random',
                  'episode_random': True}

    data_path = os.getcwd()+'/datasets/perfect_forecast/'
    results_path = os.getcwd()+'/results/battery_tests/'

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=agent_config,
                                  env=env,
                                  env_config=env_config,
                                  total_steps=total_steps,
                                  data_path=data_path,
                                  results_path=results_path,
				  seed=15,
                                  run_name='DQN_4')
