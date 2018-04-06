"""
Runs a single experiment using config dictionaries.

Command line args
    --expt_name - the directory where run results will sit
    --run_name (optional)
    --seed (optional)
"""

import os

from energy_py import experiment, make_expt_parser, make_paths, make_logger


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

    paths = make_paths(args.expt_name, run_name=args.run_name)
    logger = make_logger(paths, 'master')

    agent, env, sess = experiment(agent_config=agent_config,
                                  env_config=env_config,
                                  total_steps=total_steps,
                                  paths=paths,
                                  seed=args.seed)
