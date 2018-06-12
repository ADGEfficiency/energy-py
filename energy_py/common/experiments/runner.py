import csv
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from energy_py.common.utils import ensure_dir


class Runner(object):
    """
    Giving the runner total steps allows a percent of expt stat - very useful
    Also can control how often it logs
    """

    def __init__(self,
                 sess,
                 paths,
                 total_steps,
                 env):

        self.sess = sess
        self.rewards_path = paths['ep_rewards']
        self.tb_path = paths['tb_rl']
        self.env_hist_path = paths['env_histories']

        self.total_steps = int(total_steps)

        try:
            self.state_info = env.env.state_info
            self.observation_info = env.observation_info
        except AttributeError:
            pass

        self.log_freq = 500

        self.writer = tf.summary.FileWriter(self.tb_path, self.sess.graph)

        #  give its own logger so we can differentiate from logs in
        #  other code in this script.  maybe means this should
        #  be somewhere else TODO
        self.logger = logging.getLogger(__name__)

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
            'avg_rew_100': np.mean(self.episode_rewards[-100:]),
            'min_rew_100': np.min(self.episode_rewards[-100:]),
            'max_rew_100': np.max(self.episode_rewards[-100:]),
            'avg_rew': np.mean(self.episode_rewards),
            'min_rew': np.min(self.episode_rewards),
            'max_rew': np.max(self.episode_rewards)
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
            self.writer.add_summary(summary, self.step)
        self.writer.flush()
        with open(self.rewards_path, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.episode_rewards)

        self.current_episode_rewards = []

        if env_info:
            self.save_env_hist(env_info, len(self.episode_rewards))

    def save_env_hist(self, env_info, episode):
        """
        TODO this should maybe be in the environment????
        Saves the environment info dictionary to a csv

        args
            env_info (dict) the info dict returned from env.step()
            episode (int)
        """
        print('saving hist')
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
