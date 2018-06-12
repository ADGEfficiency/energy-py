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
                'memory_fraction': 0.15,
                'memory_type': 'deque',
                'double_q': True,
                'process_observation': 'normalizer',
                'process_target': 'standardizer'}


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


def test_flex_expt():

    env_config = {'env_id': 'flex-v0',
                  'dataset': 'example',
                  'episode_length': 10,
                  'episode_sample': 'random'}

    experiment(agent_config=AGENT_CONFIG,
               env_config=env_config,
               total_steps=TOTAL_STEPS,
               paths=PATHS)


def test_cartpole_expt():

    env_config = {'env_id': 'cartpole-v0'}

    experiment(agent_config=AGENT_CONFIG,
               env_config=env_config,
               total_steps=TOTAL_STEPS,
               paths=PATHS)


if __name__ == '__main__':
    """
    This code is here to allow debugging of the agent and environment
    in a realistic way

    Might not be up to date with experiment()
    """
    import tensorflow as tf

    from energy_py import make_paths, Runner
    from energy_py.agents import DQN
    from energy_py.envs import FlexEnv

    agent_config = {'discount': 0.97,
                    'tau': 0.001,
                    'total_steps': TOTAL_STEPS,
                    'batch_size': 32,
                    'layers': (50, 50),
                    'learning_rate': 0.0001,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.4,
                    'process_observation': False,
                    'process_target': False}

    env_config = {'episode_length': 50,
                  'episode_random': False}

    agent = DQN
    env = FlexEnv
    data_path = DATA_PATH
    results_path = RESULTS_PATH
    run_name = 'debug'

    #  start a new tensorflow session
    with tf.Session() as sess:

        #  create a dictionary of paths
        paths = make_paths(data_path, results_path, run_name)

        #  some env's don't need to be configured
        if env_config:
            env_config['data_path'] = paths['data_path']
            env = env(**env_config)

        #  add stuff into the agent config dict
        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['act_path'] = paths['tb_act']
        agent_config['learn_path'] = paths['tb_learn']

        #  init agent and save args
        agent = agent(**agent_config)

        #  runner helps to manage our experiment
        runner = Runner(tb_path=paths['tb_rl'],
                        env_hist_path=paths['env_histories'])

        #  outer while loop runs through multiple episodes
        step, episode = 0, 0
        while step < TOTAL_STEPS:
            episode += 1
            done = False
            observation = env.reset()

            #  inner while loop runs through a single episode
            while not done:
                step += 1
                #  select an action
                action = agent.act(observation)
                #  take one step through the environment
                next_observation, reward, done, info = env.step(action)
                #  store the experience
                agent.remember(observation, action, reward,
                               next_observation, done)
                #  moving to the next time step
                observation = next_observation
                runner.append(reward)

                #  fill the memory up halfway before we learn
                if step > int(agent.memory.size * 0.5):
                    train_info = agent.learn()

            runner.report({'ep': episode,
                           'step': step},
                          env_info=info)
