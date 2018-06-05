"""
Runs a single experiment using config dictionaries

Command line args
    expt_name - the directory where run results will sit
    dataset_name - name of the dataset folder in experiments/dataset
    --run_name (optional)
    --seed (optional)

To run the example experiment
    python dict_expt.py example_dict example
"""

import os

from energy_py import experiment, make_expt_parser, make_paths, make_logger


if __name__ == '__main__':
    args = make_expt_parser()
    TOTAL_STEPS = 100000

    agent_config = {'agent_id': 'DQN',
                    'discount': 0.99,
                    'tau': 0.001,
                    'total_steps': TOTAL_STEPS,
                    'batch_size': 32,
                    'layers': (25, 25, 25),
                    'learning_rate': 0.01,
                    'epsilon_decay_fraction': 0.5,
                    'memory_fraction': 0.15,
                    'memory_type': 'deque',
                    'double_q': True,
                    'process_target': 'normalizer'}

    env_config = {'env_id': 'Cartpole'}

    # env_config = {'env_id': 'Flex-v1',
    #               'dataset_name': args.dataset_name,
    #               'flex_size': 0.02,
    #               'max_flex_time': 6,
    #               'relax_time': 0,
    #               'episode_length': 96,
    #               'episode_sample': 'random'}

    # env_config = {'env_id': 'Battery',
    #               'dataset_name': args.dataset_name,
    #               'episode_sample': 'random'}

    # env_config = {'env_id': 'Flex-v0',
    #               'dataset_name': args.dataset_name,
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
