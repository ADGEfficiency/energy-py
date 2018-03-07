import logging
import os

from energy_py import experiment
from energy_py.agents import ClassifierAgent
from energy_py.envs import FlexEnv

if __name__ == '__main__':
    total_steps = 1e6


    data_path = os.getcwd()+'/classifier_deterministic/'
    results_path = os.getcwd()+'/results/classifier/'

    env_config = {'episode_length': 0,
                  'episode_random': False,
                  'data_path': data_path,
                  'flex_effy': 1.0,
                  'flex_size': 0.05}

    agent_config = {'discount': 0.9}
    total_steps = 10
    env = FlexEnv
    agent = ClassifierAgent

    agent, env, sess = experiment(agent, agent_config, env, total_steps,
                                  data_path, results_path, env_config)
