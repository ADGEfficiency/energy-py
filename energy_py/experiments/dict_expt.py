"""
Runs a single experiment using config dictionaries

Command line args
    expt_name - the directory where run results will sit
    dataset_name - name of the dataset folder in experiments/dataset
    --run_name (optional)
    --seed (optional)

To run the example experiment
    python dict_expt.py example example
"""

import os

from energy_py import experiment, make_expt_parser, make_paths, make_logger
from energy_py import get_dataset_path


if __name__ == '__main__':
    args = make_expt_parser()
    total_steps = 1e2

    agent_config = {'agent_id': 'DQN',
                    'discount': 0.99,
                    'tau': 0.001,
                    'total_steps': total_steps,
                    'batch_size': 32,
                    'layers': (25, 25, 25),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.4,
                    'memory_fraction': 0.15,
                    'memory_type': 'priority',
                    'double_q': False,
                    'process_observation': 'standardizer',
                    'process_target': 'normalizer'}

    env_config = {'env_id': 'BatteryEnv',
                  'episode_length': 2016,
                  'initial_charge': 'random',
                  'episode_random': True}

    env_config['data_path'] = get_dataset_path(args.dataset_name)

    expt_path = os.path.join(os.getcwd(),
                             'results',
                             args.expt_name)

    paths = make_paths(expt_path, run_name=args.run_name)
    logger = make_logger(paths, 'master')

    experiment(agent_config=agent_config,
               env_config=env_config,
               total_steps=total_steps,
               paths=paths,
               seed=args.seed)
