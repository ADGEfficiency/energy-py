from io import BytesIO
import json
import os
import pkg_resources

import numpy as np
import pandas as pd
import tensorflow as tf

from energypy import make_new_logger


def read_log(log_file_path):
    with open(log_file_path) as f:
        logs = f.read().splitlines()
    return [json.loads(log) for log in logs]


def load_dataset(dataset, name):
    """ load example dataset or load from user supplied path """
    if dataset == 'example':
        data = pkg_resources.resource_string(
            'energypy',
            'experiments/datasets/example/{}.csv'.format(name)
        )

        return pd.read_csv(
            BytesIO(data), index_col=0, parse_dates=True
        )

    else:
        return pd.read_csv(
            os.path.join(dataset, name + '.csv'), index_col=0, parse_dates=True
        )


class Runner():
    """ logs episode reward stats """

    def __init__(self, sess, run_cfg):

        self.writer = tf.summary.FileWriter(
            run_cfg['tensorboard_dir'], sess.graph
        )

        self.logger = make_new_logger('results', run_cfg['run_dir'])

        self.reset()

    def reset(self):
        self.history = []
        self.step = 0

    def record_episode(self, episode_rewards):

        total_episode_reward = np.sum(episode_rewards)
        self.history.append(total_episode_reward)

        summaries = {
            'total_episode_reward': total_episode_reward,
            'avg_rew_100': np.mean(self.history[-100:]),
            'min_rew_100': np.min(self.history[-100:]),
            'max_rew_100': np.max(self.history[-100:]),
            'avg_rew': np.mean(self.history),
            'min_rew': np.min(self.history),
            'max_rew': np.max(self.history)
        }

        for tag, value in summaries.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=float(value))])
            self.writer.add_summary(summary, self.step)

        self.writer.flush()

        log = {
            'episode': len(self.history), 
            'reward': total_episode_reward, 
            'rew_100': summaries['avg_rew_100']
        }

        self.logger.info(json.dumps(log))
