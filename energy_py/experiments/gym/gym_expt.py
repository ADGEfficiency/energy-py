import logging

import gym

from energy_py import experiment
from energy_py.agents import DQN


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

    envs = ['Pendulum-v0', 'CartPole-v1', 'MountainCar-v0']
    env = gym.make(envs[1])
    total_steps = 1e5
    base_path = './dqn'
    info = experiment(agent, agent_config, env,
                      total_steps, base_path)
