import logging
import os

from energy_py import experiment
from energy_py.agents import DQN
from energy_py.envs import CartPoleEnv

if __name__ == '__main__':

    total_steps = 1e5
    agent = DQN
    agent_config = {'discount': 0.97,
                    'tau': 0.001,
                    'total_steps': 500000,
                    'batch_size': 32,
                    'layers': (50, 50),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.1,
                    'memory_type': 'priority',
                    'total_steps': total_steps}

    env = CartPoleEnv()

    base_path = './gym/dqn'

    data_path = os.getcwd()+'/gym/'
    results_path = os.getcwd()+'/results/gym/'

    agent, env, sess = experiment(agent=DQN, 
                                  agent_config=agent_config, 
                                  env=env,
                                  total_steps=total_steps, 
                                  data_path=data_path, 
                                  results_path=results_path)
