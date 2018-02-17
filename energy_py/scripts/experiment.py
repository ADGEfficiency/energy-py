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
import logging
import logging.config
import os
import os.path.join as join
import time

import pandas as pd
import tensorflow as tf

from energy_py import save_args, ensure_dir, make_logger, TensorboardHepler


def make_paths(data_path, results_path):
    """
    Creates a dictionary of paths for use with experiments

    args
        data_path (str) location of state.csv, observation.csv
        results_path (str)
    """
    paths = {'data_path': join(data_path),
             'results_path': join(results_path),

             'tb_rl': join(results_path, 'tensorboard', 'rl'),
             'tb_act': join(results_path, 'tensorboard', 'act'),
             'tb_learn': join(results_path, 'tensorboard', 'learn'),

             'logs': join(results_path, 'logs.log'),
             'env_args': join(results_path, 'env_args.txt'),
             'agent_args': join(results_path, 'agent_args.txt'),
             'env_histories': join(results_path, 'env_histories')}

    for key, path in paths.items():
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
