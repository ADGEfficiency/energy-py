"""
Runs a single experiment using config dictionaries

Command line args
    expt_name - the directory where run results will sit
    dataset_name - name of the dataset folder in experiments/dataset
    --run_name (optional)
    --seed (optional)

To run the example experiment, saving results into example/run0
    python dict_expt.py example_dict example run0
"""

import os

from energy_py import experiment
from energy_py.common.experiments.utils import make_expt_parser, make_paths
from energy_py.common.utils import make_logger


if __name__ == '__main__':
    args = make_expt_parser()
    TOTAL_STEPS = 400000

    agent_config = {
        'agent_id': 'naive_flex',
        'hours': (6, 11, 15, 19)
    }

    # agent_config = {
    #     'agent_id': 'dqn',
    #     'discount': 0.99,
    #     'tau': 0.001,
    #     'total_steps': TOTAL_STEPS,
    #     'batch_size': 32,
    #     'layers': (25, 25, 25),
    #     'epsilon_decay_fraction': 0.5,
    #     'memory_fraction': 0.15,
    #     'memory_type': 'deque',
    #     'double_q': False,
    #     'learning_rate': 0.0001,
    #     'decay_learning_rate': 0.1
    #                 }

    env_config = {'env_id': 'flex-v0',
                  'dataset': 'tempus', 
                  'flex_size': 0.5,
                  'flex_time': 1,
                  'relax_time': 0,
                  'episode_length': 2016,
                  'episode_sample': 'random'}

    # env_config = {'env_id': 'CartPole'}

    # env_config = {'env_id': 'Battery',
    #               'dataset_name': args.dataset,
    #               'episode_sample': 'random'}

    # env_config = {'env_id': 'Flex-v0',
    #               'dataset_name': args.dataset,
    #               'episode_sample': 'random'}

    expt_path = os.path.join(os.getcwd(),
                             'results',
                             args.expt_name)

    paths = make_paths(expt_path, run_name=args.run_name)
    logger = make_logger(paths, 'master')

    experiment(agent_config=agent_config,
               env_config=env_config,
               total_steps=TOTAL_STEPS,
               paths=paths,
               seed=args.seed)
