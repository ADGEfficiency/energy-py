import os

from energy_py import experiment, make_expt_parser
from energy_py.agents import DQN, DPG
from energy_py.envs import FlexEnv, BatteryEnv

if __name__ == '__main__':
    args = make_expt_parser()

    total_steps = 1e2
    agent_config = {'discount': 0.99,
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

    env = BatteryEnv
    env_config = {'episode_length': 2016,
                  'initial_charge': 'random',
                  'episode_random': True}

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=agent_config,
                                  env=env,
                                  total_steps=total_steps,
                                  paths=paths,
                                  seed=args.seed,
                                  run_name=args.name)
