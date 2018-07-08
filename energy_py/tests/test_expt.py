"""
Runs a simple experiment to test nothing is broken

The following tests are run, all with the DQN agent

test_battery_expt()
test_flex_expt()
test_cartpole_expt()
"""
import os

import energy_py
from energy_py import experiment, get_dataset_path
from energy_py.common.experiments.utils import make_paths

TOTAL_STEPS = 200

PATHS = make_paths('results')

AGENT_CONFIG = {'agent_id': 'dqn',
                'discount': 0.97,
                'tau': 0.001,
                'total_steps': TOTAL_STEPS,
                'batch_size': 32,
                'layers': (25, 25, 25),
                'learning_rate': 0.0001,
                'epsilon_decay_fraction': 0.3,
                'memory_type': 'deque',
                'double_q': True}


def test_battery_expt():

    env_config = {'env_id': 'battery',
                  'dataset': 'example',
                  'episode_length': 10,
                  'episode_sample': 'random',
                  'initial_charge': 'random'}

    experiment(agent_config=AGENT_CONFIG,
               env_config=env_config,
               total_steps=TOTAL_STEPS,
               paths=PATHS)


# def test_flex_expt():

#     env_config = {'env_id': 'flex-v0',
#                   'dataset': 'example',
#                   'episode_length': 10,
#                   'episode_sample': 'random'}

#     experiment(agent_config=AGENT_CONFIG,
#                env_config=env_config,
#                total_steps=TOTAL_STEPS,
#                paths=PATHS)


def test_cartpole_expt():

    env_config = {'env_id': 'cartpole-v0'}

    experiment(agent_config=AGENT_CONFIG,
               env_config=env_config,
               total_steps=TOTAL_STEPS,
               paths=PATHS)
