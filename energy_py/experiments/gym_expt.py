import logging
import os

from energy_py import experiment
from energy_py.agents import DPG, DQN
from energy_py.envs import CartPoleEnv, PendulumEnv

if __name__ == '__main__':

    total_steps = 1e5
    agent = DQN
    agent_config = {'discount': 0.97,
                    'tau': 0.001,
                    'batch_size': 32,
                    'layers': (10, 10, 10),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.2,
                    'memory_type': 'deque',
                    'double_q': False,
                    'total_steps': total_steps,
                    'target_processor': 'normalizer',
                    'observation_processor': 'standardizer'}

    env = CartPoleEnv()

    results_path = os.getcwd()+'/results/cartpole_tests/'

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=agent_config,
                                  env=env,
                                  total_steps=total_steps,
                                  results_path=results_path,
                                  run_name='DQN_1')
