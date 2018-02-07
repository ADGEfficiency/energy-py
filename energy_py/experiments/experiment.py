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
import time

import numpy as np
import tensorflow as tf

from energy_py import ensure_dir


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


def make_paths(name):
    results = name + '/'
    paths = {'results': results,
             'brain': results + 'brain/',
             'tb_rl': results + 'tb/rl/',
             'tb_act': results + 'tb/act/',
             'tb_learn': results + 'tb/learn/',
             'logs': results + 'logs.log',
             'args': results + 'args.txt',
             'env_args': results + 'env_args.txt',
             'agent_args': results + 'agent_args.txt'}
    for k, path in paths.items():
        ensure_dir(path)
    return paths


def make_logger(log_path, log_status='INFO'):

    logger = logging.getLogger(__name__)

    logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,

            'formatters': {'standard': {'format': '%(asctime)s [%(levelname)s]%(name)s: %(message)s'}},

            'handlers': {'console': {'level': log_status,
                                     'class': 'logging.StreamHandler',
                                     'formatter': 'standard'},

                         'file': {'class': 'logging.FileHandler',
                                  'level': 'DEBUG',
                                  'filename': log_path,
                                  'mode': 'w',
                                  'formatter': 'standard'}, },

            'loggers': {'': {'handlers': ['console', 'file', ],
                             'level': 'DEBUG',
                             'propagate': True}}})

    return logger


def experiment(agent, agent_config, env, total_steps, base_path):
    """
    Run an experiment.  Episodes are run until total_steps are reached.

    args
        agent (object) learner & decision maker
        agent_config (dict)
        env (object) reinforcment learning environment
        total_steps (int) length of the experiment
        base_path (str) used to setup results folders

    returns
        global_rewards (list) rewards for each episode

    """

    with tf.Session() as sess:
        paths = make_paths(base_path)

        logger = make_logger(paths['logs'], 'INFO')

        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['act_path'] = paths['tb_act']
        agent_config['learn_path'] = paths['tb_learn']
        agent = agent(**agent_config)

        save_args(agent_config, path=paths['agent_args'])

        runner = Runner(sess, paths['tb_rl'])

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
            #  reporting expt status at the end of each episode
            runner.report({'ep': episode,
                          'ep_rew': sum(rewards),
                          'avg_rew': avg_rew})

    return global_rewards


class Runner(object):
    """
    Trying to figure out what to do here - trying this runner class
    """
    def __init__(self, sess, logdir):
        self.sess = sess

        self.start_time = time.time()
        self.logger_timer = logging.getLogger('runner')

        self.writer = tf.summary.FileWriter(logdir)
        self.steps = 0

    def calc_time(self):
        return (time.time() - self.start_time) / 60

    def report(self, summaries):
        """
        The main functionality of this class
        """
        summaries['run_time'] = self.calc_time()
        log = ['{} : {:.2f}'.format(k, v) for k, v in summaries.items()]
        self.logger_timer.info(log)

        self.steps += 1
        no_tb = ['ep', 'run_time']
        for tag, var in summaries.items():
            if tag in no_tb:
                pass
            else:
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                             simple_value=var)])
                self.writer.add_summary(summary, self.steps)

        self.writer.flush()

