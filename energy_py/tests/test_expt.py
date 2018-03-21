"""
Runs a simple experiment to test nothing is broken

The following tests are run, all with the DQN agent

test_battery_expt()
test_flex_expt()
test_cartpole_expt()
"""
import os

from energy_py.agents import DQN
from energy_py.envs import BatteryEnv, CartPoleEnv, FlexEnv
from energy_py import experiment


DATA_PATH = os.getcwd()+'/data/'
RESULTS_PATH = os.getcwd()+'/results/'
TOTAL_STEPS = 200

AGENT_CONFIG = {'discount': 0.97,
                'tau': 0.001,
                'total_steps': TOTAL_STEPS,
                'batch_size': 32,
                'layers': (25, 25, 25),
                'learning_rate': 0.0001,
                'epsilon_decay_fraction': 0.3,
                'memory_fraction': 0.15,
                'memory_type': 'priority',
                'double_q': True,
                'process_observation': 'normalizer',
                'process_target': 'standardizer'}


def test_battery_expt():

    env = BatteryEnv

    env_config = {'episode_length': 10,
                  'episode_random': True,
                  'initial_charge': 'random'}

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=AGENT_CONFIG,
                                  env=env,
                                  env_config=env_config,
                                  total_steps=TOTAL_STEPS,
                                  data_path=DATA_PATH,
                                  results_path=RESULTS_PATH)


def test_flex_expt():

    env = FlexEnv

    env_config = {'episode_length': 10,
                  'episode_random': True}

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=AGENT_CONFIG,
                                  env=env,
                                  env_config=env_config,
                                  total_steps=TOTAL_STEPS,
                                  data_path=DATA_PATH,
                                  results_path=RESULTS_PATH)

def test_cartpole_expt():

    env = CartPoleEnv()

    agent, env, sess = experiment(agent=DQN,
                                  agent_config=AGENT_CONFIG,
                                  env=env,
                                  total_steps=TOTAL_STEPS,
                                  data_path=DATA_PATH,
                                  results_path=RESULTS_PATH)
