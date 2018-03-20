"""
A collection of functions to run experiments.

Module contains:
    expt_args - creates and parses command line arguments
    save_args - saves dictionaries or argparses to text files
    make_paths - creates a dictionary of paths
    make_logger - DEBUG to file, INFO to console
    experiment - runs a single reinforcment learning experiment
    Runner - class to save environment data & TensorBoard
"""

import datetime
import logging
import logging.config
import os
import pdb
import time

import pandas as pd
import tensorflow as tf

from energy_py import save_args, ensure_dir, make_logger, TensorboardHepler


def make_paths(data_path, results_path, run_name=None):
    """
    Creates a dictionary of paths for use with experiments

    args
        data_path (str) location of state.csv, observation.csv
        results_path (str)
        run_name (str) optional name for the tensorboard run
    """
    #  use a timestamp if no run_name is supplied
    if run_name is None:
        run_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    #  rename the join function to make code below eaiser to read
    join = os.path.join

    #  run_path is the folder where output from this run will be saved in
    run_path = join(results_path, run_name)

    paths = {'data_path': data_path,
             'run_path': run_path,

             #  tensorboard runs are all in the tensoboard folder
             #  this is for easy comparision of run
             'tb_rl': join(results_path, 'tensorboard', run_name, 'rl'),
             'tb_act': join(results_path, 'tensorboard', run_name, 'act'),
             'tb_learn': join(results_path, 'tensorboard', run_name,  'learn'),
             'env_histories': join(run_path, 'env_histories'),

             #  run specific folders are in another folder
             'debug_log': join(run_path, 'debug.log'),
             'info_log': join(run_path, 'info.log'),
             'env_args': join(run_path, 'env_args.txt'),
             'agent_args': join(run_path, 'agent_args.txt')}

    #  check that all our paths exist
    for key, path in paths.items():
        ensure_dir(path)

    return paths


def experiment(agent, agent_config, env,
               total_steps, data_path, results_path,
               run_name=None, env_config=None, seed=None):
    """
    Run an experiment.  Episodes are run until total_steps are reached.

    args
        agent (object) learner & decision maker
        agent_config (dict)
        env (object) reinforcment learning environment
        env_config (dict)
        total_steps (int)
        data_path (str)
        results_path (str)
        seed (int)

    returns
        agent (object)
        env (object)
        sess (tf.Session)
    """
    #  start a new tensorflow session
    with tf.Session() as sess:

        #  optionally set random seeds
        if seed:
            tf.set_random_seed(seed)
            np.random.seed(seed)

        #  create a dictionary of paths
        paths = make_paths(data_path, results_path, run_name)

        #  some env's don't need to be configured
        if env_config:
            env_config['data_path'] = paths['data_path']
            env = env(**env_config)
            save_args(env_config, path=paths['env_args'])

        #  setup the logging config
        logger = make_logger(paths, name='experiment')

        #  add stuff into the agent config dict
        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['act_path'] = paths['tb_act']
        agent_config['learn_path'] = paths['tb_learn']

        #  init agent and save args
        agent = agent(**agent_config)
        save_args(agent_config, path=paths['agent_args'])

        #  runner helps to manage our experiment
        runner = Runner(tb_path=paths['tb_rl'],
                        env_hist_path=paths['env_histories'])

        #  outer while loop runs through multiple episodes
        step, episode = 0, 0
        while step < total_steps:
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

    return agent, env, sess


class Runner(object):
    """
    Class to help run experiments.

    Currently performs three roles
        keeping track of rewards
        keeping track of run time
        processing environment history

    args
        tb_path (str)  path where tb logs sit
        env_hist_path (str)  path to save env data too
    """
    def __init__(self, tb_path=None, env_hist_path=None):

        self.start_time = time.time()
        self.logger_timer = logging.getLogger('runner')

        if tb_path:
            self.tb_helper = TensorboardHepler(tb_path)

        if env_hist_path:
            self.env_hist_path = env_hist_path

        #  a list to hold the rewards for a single episode
        self.ep_rewards = []
        #  a list to hold rewards for all episodes
        self.global_rewards = []

    def append(self, reward):
        self.ep_rewards.append(reward)

    def calc_time(self):
        return (time.time() - self.start_time) / 60

    def report(self, summaries, env_info=None):
        """
        The main functionality of this class

        Should be run at the end of each episode
        """

        if env_info:
            output = pd.DataFrame().from_dict(env_info)

            csv_path = os.path.join(self.env_hist_path,
                                    'ep_{}'.format(summaries['ep']),
                                    'hist.csv')
            ensure_dir(csv_path)
            output.to_csv(csv_path)

        #  now episode has finished, we save our rewards onto our global list
        self.global_rewards.append(sum(self.ep_rewards))

        self.avg_rew = sum(self.global_rewards[-100:]) / len(self.global_rewards[-100:])

        summaries['ep_rew'] = sum(self.ep_rewards)
        summaries['avg_rew'] = self.avg_rew

        #  add the run time so we can log the summaries
        summaries['run_time'] = self.calc_time()
        log = ['{} : {:.2f}'.format(k, v) for k, v in summaries.items()]
        self.logger_timer.info(log)

        if hasattr(self, 'tb_helper'):
            no_tb = ['ep', 'run_time', 'step']
            _ = [summaries.pop(key) for key in no_tb]
            self.tb_helper.add_summaries(summaries)

        #  reset the counter for episode rewards
        self.ep_rewards = []


if __name__ == '__main__':
    """
    This code is here to allow debugging of the agent and environment
    in a realistic way

    Might not be up to date with experiment()
    """
    from energy_py.agents import DQN
    from energy_py.envs import CartPoleEnv

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

    agent = DQN
    total_steps = 1000

    with tf.Session() as sess:
        env = CartPoleEnv()

        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['total_steps'] = total_steps
        agent = agent(**agent_config)

        runner = Runner()
        step, episode = 0, 0
        global_rewards = []

        #  outer while loop runs through multiple episodes
        step = 0
        while step < total_steps:
            episode += 1
            done = False
            observation = env.reset()
            rewards = []
            #  inner while loop runs through a single episode
            while not done:
                step += 1

                #  select an action
                action = agent.act(observation)
                #  take one step through the environment
                next_observation, reward, done, info = env.step(action[0])
                #  store the experience
                agent.remember(observation, action, reward,
                               next_observation, done)
                #  moving to the next time step
                observation = next_observation
                rewards.append(reward)

                print(step)
                if step > agent.memory.size * 0.5:
                    print('LEARNING')
                    train_info = agent.learn()

            global_rewards.append(sum(rewards))
            avg_rew = sum(global_rewards[-100:]) / len(global_rewards[-100:])

            runner.report({'ep': episode,
                           'step': step,
                           'ep_rew': sum(rewards),
                           'avg_rew': avg_rew})


