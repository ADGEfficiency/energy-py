"""
the Runner class has two main functions
1 - keeping track of episode rewards
2 - logging reward info to tensorboard
3 - saving reward history to csv
"""


import csv
import logging

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class Runner(object):
    """
    Giving the runner total steps allows a percent of expt stat - very useful
    Also can control how often it logs
    """

    def __init__(
            self,
            sess,
            paths,
            total_steps
    ):

        self.sess = sess
        self.rewards_path = paths['ep_rewards']
        self.tb_path = paths['tb_rl']
        self.total_steps = int(total_steps)

        self.writer = tf.summary.FileWriter(
            self.tb_path, self.sess.graph
        )

        self.log_freq = 500
        logging.info('Making runner - log every {} steps'.format(
            self.log_freq))

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

        log_string = 'Episode {:0.0f} step {:0.0f} {:2.1f}%'.format(
            len(self.episode_rewards),
            self.step,
            100 * self.step / self.total_steps
        )

        logger.debug(log_string)
        [logger.debug('{} - {}'.format(k, v)) for k, v in summaries.items()]

        if self.step % self.log_freq == 0:
            logger.info(log_string)
            logger.info('{} - {:2.1f}'.format(
                'avg_rew_100', summaries['avg_rew_100']
            )
                        )

        for tag, value in summaries.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=float(value))])
            self.writer.add_summary(summary, self.step)

        self.writer.flush()

        with open(self.rewards_path, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.episode_rewards)

        self.current_episode_rewards = []
