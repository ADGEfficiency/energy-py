"""
A collection of functions to run experiments.

Module contains:
    make_expt_parser - parses command line arguments for experiments
    make_paths - creates a dictionary of paths
    run_config_expt - runs an experiment using a config file
    experiment - runs a single reinforcment learning experiment
    Runner - class to save environment data & TensorBoard
"""

import argparse
import datetime
import csv
import logging
import logging.config
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import energy_py
from energy_py.scripts.utils import save_args, ensure_dir, make_logger
from energy_py.scripts.utils import parse_ini
from energy_py.experiments.datasets import get_dataset_path


logger = logging.getLogger(__name__)


def make_expt_parser():
    """
    Parses arguments from the command line for running experiments

    returns
        args (argparse NameSpace)
    """
    parser = argparse.ArgumentParser(description='energy_py experiment argparser')

    #  required
    parser.add_argument('expt_name', default=None, type=str)
    parser.add_argument('dataset_name', default=None, type=str)
    #  optional
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()

    return args


def make_paths(expt_path, run_name=None):
    """
    Creates a dictionary of paths for use with experiments

    args
        expt_path (str)
        run_name (str) optional name for run.  Timestamp used if not given

    returns
        paths (dict) {name: path}

    Folder structure
        experiments/results/expt_name/run_name/tensoboard/run_name/rl
                                                                  /act
                                                                  /learn
                                               env_histories/ep_1/hist.csv
                                                             ep_2/hist.csv
                                                             e..
                                               common.ini
                                               run_configs.ini
                                               agent_args.txt
                                               env_args.txt
                                               info.log
                                               debug.log
    """
    #  use a timestamp if no run_name is supplied
    if run_name is None:
        run_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    #  rename the join function to make code below eaiser to read
    join = os.path.join

    #  run_path is the folder where output from this run will be saved in
    run_path = join(expt_path, run_name)

    paths = {'run_path': run_path,

             #  config files
             'common_config': join(expt_path, 'common.ini'),
             'run_configs': join(expt_path, 'run_configs.ini'),

             #  tensorboard runs are all in the tensoboard folder
             #  this is for easy comparision of run
             'tb_rl': join(expt_path, 'tensorboard', run_name, 'rl'),
             'tb_act': join(expt_path, 'tensorboard', run_name, 'act'),
             'tb_learn': join(expt_path, 'tensorboard', run_name,  'learn'),
             'env_histories': join(run_path, 'env_histories'),

             #  run specific folders are in another folder
             'debug_log': join(run_path, 'debug.log'),
             'info_log': join(run_path, 'info.log'),
             'env_args': join(run_path, 'env_args.txt'),
             'agent_args': join(run_path, 'agent_args.txt'),
             'ep_rewards': join(run_path, 'ep_rewards.csv')}

    #  check that all our paths exist
    for key, path in paths.items():
        ensure_dir(path)

    return paths


def run_config_expt(expt_name, run_name, expt_path):
    """
    Runs a single experiment, reading experiment setup from a .ini

    args
        expt_name (str)
        run_name (str)
        expt_path (str)

    Each experiment is made of multiple runs.  This function will load one run
    and run an experiment.
    """

    paths = make_paths(expt_path, run_name=run_name)
    logger = make_logger(paths, 'master')

    env_config = parse_ini(paths['common_config'], 'env')
    env_config['data_path'] = get_dataset_path(env_config['dataset_name'])
    env_config.pop('dataset_name')

    agent_config = parse_ini(paths['run_configs'], run_name)

    experiment(agent_config,
               env_config,
               agent_config['total_steps'],
               paths,
               seed=agent_config['seed'])


def experiment(agent_config,
               env_config,
               total_steps,
               paths,
               seed=None):
    """
    Run an experiment.  Episodes are run until total_steps are reached.

    args
        agent_config (dict)
        env_config (dict)
        total_steps (int)
        paths (dict)
        seed (int)

    Agent and environment are created from config dictionaries.
    """
    tf.reset_default_graph()
    with tf.Session() as sess:

        #  optionally set random seeds
        logger.info('random seed is {}'.format(seed))
        if seed:
            seed = int(seed)
            random.seed(seed)
            tf.set_random_seed(seed)
            np.random.seed(seed)

        env = energy_py.make_env(**env_config)
        save_args(env_config, path=paths['env_args'])

        #  add stuff into the agent config dict
        agent_config['env'] = env
        agent_config['env_repr'] = repr(env)
        agent_config['sess'] = sess
        agent_config['act_path'] = paths['tb_act']
        agent_config['learn_path'] = paths['tb_learn']

        #  init agent and save args
        agent = energy_py.make_agent(**agent_config)
        if hasattr(agent, 'acting_writer'):
            agent.acting_writer.add_graph(sess.graph)
        save_args(agent_config, path=paths['agent_args'])

        #  runner helps to manage our experiment
        runner = Runner(sess, paths)

        #  outer while loop runs through multiple episodes
        step, episode = 0, 0
        while step < int(total_steps):
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
                runner.record_step(reward)
                #  moving to the next time step
                observation = next_observation

                #  fill the memory up halfway before we learn
                #  TODO the agent should decide what to do internally here
                if step > int(agent.memory.size * 0.5):
                    train_info = agent.learn()

            runner.record_episode(env_info=info)


class Runner(object):
    """
    Giving the runner total steps allows a percent of expt stat - very useful
    Also can control how often it logs
    """

    def __init__(self,
                 sess,
                 paths,
                 total_steps):

        self.sess = sess
        self.rewards_path = paths['ep_rewards']
        self.tb_path = paths['tb_rl']
        self.total_steps = total_steps
        self.log_freq = 500 

        self.writer = tf.summary.FileWriter(self.tb_path, self.sess.graph)

        #  give its own logger so we can differentiate from logs in 
        #  other code in this script.  maybe means this should 
        #  be somewhere else TODO
        self.logger = logging.getLogger('runner')

        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.step = 0

    def record_step(self, reward):

        self.current_episode_rewards.append(reward)
        self.step += 1

    def record_episode(self, env_info=None):

        total_episode_reward = sum(self.current_episode_rewards)
        self.episode_rewards.append(total_episode_reward)

        summaries = {
            'total_episode_reward': total_episode_reward,
            'avg_rew': np.mean(self.episode_rewards[-50:]),
            'min_rew': np.min(self.episode_rewards[-50:]),
            'max_rew': np.max(self.episode_rewards[-50:])
        }
        log_string = 'Episode {} step {} {}%'.format(
            len(self.episode_rewards),
            self.step,
            self.total_steps / self.step
        )
        #  repeated code here! TODO
        self.logger.debug(log_string)
        [self.logger.debug('{} - {}'.format(k, v)) for k, v in summaries.items()]

        if self.step % self.log_freq == 0:
            self.logger.info(log_string)
            [self.logger.info('{} - {}'.format(k, v)) for k, v in summaries.items()]

        for tag, value in summaries.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=float(value))]
            )
            self.writer.add_summary(summary, len(self.episode_rewards))
        self.writer.flush()
        with open(self.rewards_path, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.episode_rewards)

        self.current_episode_rewards = []

        # if env_info:
        #     self.save_env_hist(env_info, len(self.episode_rewards))

    def save_env_hist(self, env_info, episode):
        """
        TODO this should maybe be in the environment????
        Saves the environment info dictionary to a csv

        args
            env_info (dict) the info dict returned from env.step()
            episode (int)
        """
        output = []

        for key, info in env_info.items():

           if isinstance(info[0], np.ndarray):
               df = pd.DataFrame(np.array(info).reshape(len(info), -1))

               if key == 'observation' and self.observation_info:
                   df.columns = ['{}_{}'.format(key, o)
                                 for o in self.observation_info]

               elif key == 'next_observation' and self.observation_info:
                   df.columns = ['{}_{}'.format(key, o)
                                 for o in self.observation_info]

               elif key == 'state' and self.state_info:
                   df.columns = ['{}_{}'.format(key, s)
                                 for s in self.state_info]

               elif key == 'next_state' and self.state_info:
                   df.columns = ['{}_{}'.format(key, s)
                                 for s in self.state_info]

               else:
                   df.columns = ['{}_{}'.format(key, n)
                                 for n in range(df.shape[1])]

           else:
               df = pd.DataFrame(info, columns=[key])

           output.append(df)

        output = pd.concat(output, axis=1)

        csv_path = os.path.join(self.env_hist_path,
                               'ep_{}'.format(episode),
                               'hist.csv')
        ensure_dir(csv_path)
        output.to_csv(csv_path)



#class Runner(object):
#    """
#    Class to help run experiments.

#    args
#        tb_path (str)  path where tb logs sit
#        env_hist_path (str)  path to save env data too

#    Currently performs three roles
#        keeping track of rewards and writing to TensorBoard
#        keeping track of run time
#        processing environment history into hist.csv
#    """
#    def __init__(self,
#                 rewards_path=None,
#                 tb_path=None,
#                 env_hist_path=None,
#                 state_info=None,
#                 observation_info=None):

#        self.state_info = state_info
#        self.observation_info = observation_info

#        self.start_time = time.time()
#        self.logger_timer = logging.getLogger('runner')

#        if rewards_path:
#            self.rewards_path = rewards_path

#        if tb_path:
#            self.tb_helper = TensorboardHepler(tb_path)

#        if env_hist_path:
#            self.env_hist_path = env_hist_path

#        #  a list to hold the rewards for a single episode
#        self.ep_rewards = []
#        #  a list to hold rewards for all episodes
#        self.global_rewards = []

#    def append(self, reward):
#        self.ep_rewards.append(reward)

#    def calc_time(self):
#        return (time.time() - self.start_time) / 60

#    def report(self, summaries, env_info=None):
#        """
#        The main functionality of this class

#        Should be run at the end of each episode
#        """
#        #  now episode has finished, we save our rewards onto our global list
#        self.global_rewards.append(sum(self.ep_rewards))
#        self.avg_rew = sum(self.global_rewards[-100:]) / len(self.global_rewards[-100:])

#        #  save the reward statisistics into the summary dictionary
#        summaries['ep_rew'] = sum(self.ep_rewards)
#        summaries['avg_rew'] = self.avg_rew

#        #  add the run time so we can log the summaries
#        summaries['run_time'] = self.calc_time()
#        log = ['{} : {:02.1f}'.format(k, v) for k, v in summaries.items()]
#        self.logger_timer.info(log)

#        #  save the environment info dictionary to a csv
#        if env_info:
#            self.save_env_hist(env_info, summaries['ep'])

#        #  send the summaries to TensorBoard
#        if hasattr(self, 'tb_helper'):
#            no_tb = ['ep', 'run_time', 'step']
#            _ = [summaries.pop(key) for key in no_tb]
#            self.tb_helper.add_summaries(summaries)

#        #  reset the counter for episode rewards
#        self.ep_rewards = []

#    def save_rewards(self):
#        """
#        Saves the global rewards list to a csv
#        """
#        with open(self.rewards_path, 'w') as myfile:
#            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#            wr.writerow(self.global_rewards)

