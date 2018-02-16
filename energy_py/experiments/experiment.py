"""
A collection of functions to run experiments.

Module contains:
    expt_args - creates and parses command line arguments
    save_args - saves dictionaries or argparses to text files
    make_paths - creates a dictionary of paths
    make_logger - DEBUG to file, INFO to console
    experiment - runs a single reinforcment learning experiment
    Timer - times experiments
"""
import argparse
import csv
import logging
import logging.config
import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf

from energy_py import ensure_dir, make_logger, TensorboardHepler


def expt_args(optional_args=None):
    """
    args
        optional_args (list) one dict per optional parser arg required
    """
    parser = argparse.ArgumentParser(description='energy_py expt arg parser')

    args_list = [{'name': '--ep',
                  'type': int,
                  'default': 10,
                  'help': 'number of episodes to run (default: 10)'},
                 {'name': '--len',
                  'type': int,
                  'default': 48,
                  'help': 'length of a single episode (default: 48)'},
                 {'name': '--rand',
                  'type': bool,
                  'default': False,
                  'help': 'randomize start of each ep (default: False)'},
                 {'name': '--gamma',
                  'type': float,
                  'default': 0.9,
                  'help': 'discount rate (default: 0.9)'},
                 {'name': '--out',
                  'type': int,
                  'default': 500,
                  'help': 'output results every n episodes (default: n=500)'},
                 {'name': '--log',
                  'type': str,
                  'default': 'INFO',
                  'help': 'logging status (default: info)'}]

    if optional_args:
        args_list.append(optional_args)

    for arg in args_list:
        parser.add_argument(arg['name'],
                            type=arg['type'],
                            default=arg['default'],
                            help=arg['help'])

    args = parser.parse_args()

    return parser, args


def save_args(config, path, argparse=None):
    """
    Saves a config dictionary and optional argparse object to a text file.

    args
        config (dict)
        path (str) path for output text file
        argparse (object)
    returns
        writer (object) csv Writer object
    """
    with open(path, 'w') as outfile:
        writer = csv.writer(outfile)

        for k, v in config.items():
            print('{} : {}'.format(k, v))
            writer.writerow([k]+[v])

        if argparse:
            for k, v in vars(argparse).items():
                print('{} : {}'.format(k, v))
                writer.writerow([k]+[v])

    return writer


def make_paths(data_path, results_path):

    paths = {'data_path': data_path,
             'results': results_path,
             'tb_rl': results_path + '/tensorboard/rl/',
             'tb_act': results_path + '/tensorboard/act/',
             'tb_learn': results_path + '/tensorboard/learn/',
             'logs': results_path + 'logs.log',
             'env_args': results_path + 'env_args.txt',
             'agent_args': results_path + 'agent_args.txt',
             'env_histories': results_path + '/env_histories/'}

    for k, path in paths.items():
        ensure_dir(path)

    return paths


def experiment(agent, agent_config, env, env_config,
               total_steps, data_path, results_path):
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

    returns

    """

    with tf.Session() as sess:
        paths = make_paths(data_path, results_path)

        env_config['data_path'] = paths['data_path']
        env = env(**env_config)

        logger = make_logger(paths['logs'], 'INFO')

        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['act_path'] = paths['tb_act']
        agent_config['learn_path'] = paths['tb_learn']
        agent = agent(**agent_config)

        save_args(agent_config, path=paths['agent_args'])
        save_args(env_config, path=paths['env_args'])

        runner = Runner(paths)
        step, episode = 0, 0
        global_rewards = []

        #  outer while loop runs through multiple episodes
        while step < total_steps:
            episode += 1
            done, step = False, 0
            observation = env.reset()
            rewards = []
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
                rewards.append(reward)

                if step > agent.initial_random:
                    train_info = agent.learn()

            global_rewards.append(sum(rewards))
            avg_rew = sum(global_rewards[-100:]) / len(global_rewards[-100:])

            runner.report({'ep': episode,
                           'step': step,
                           'ep_rew': sum(rewards),
                           'avg_rew': avg_rew},
                          env_info=info)

    return global_rewards


class Runner(object):
    """
    Trying to figure out what to do here - trying this runner class
    """
    def __init__(self, paths):

        self.start_time = time.time()
        self.logger_timer = logging.getLogger('runner')

        self.tb_helper = TensorboardHepler(paths['tb_rl'])
        self.env_hist_path = paths['env_histories']

    def calc_time(self):
        return (time.time() - self.start_time) / 60

    def report(self, summaries, env_info=None):
        """
        The main functionality of this class
        """
        summaries['run_time'] = self.calc_time()
        log = ['{} : {:.2f}'.format(k, v) for k, v in summaries.items()]
        self.logger_timer.info(log)

        if env_info:
            output = pd.DataFrame().from_dict(env_info)
            output.set_index('steps', drop=True)

            csv_path = os.path.join(self.env_hist_path,
                                    'ep_{}'.format(summaries['ep']),
                                    'hist.csv')
            ensure_dir(csv_path)
            output.to_csv(csv_path)

        no_tb = ['ep', 'run_time', 'step']
        _ = [summaries.pop(key) for key in no_tb]
        self.tb_helper.add_summaries(summaries)

