import numpy as np
import tensorflow as tf

from energy_py import ensure_dir


class EpisodeStats(object):

    def __init__(self,
                 sess,
                 tb_dir):

        self.sess = sess

        ensure_dir(tb_dir)
        self.writer = tf.summary.FileWriter(tb_dir, self.sess.graph)

        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.current_episode_rewards = []

    def record_step(self, reward):

        self.current_episode_rewards.append(reward)

    def record_episode(self):

        total_episode_reward = sum(self.current_episode_rewards)
        self.episode_rewards.append(total_episode_reward)

        summaries = {
            'total_episode_reward': total_episode_reward,
            'avg_rew': np.mean(self.episode_rewards[-50:]),
            'min_rew': np.min(self.episode_rewards[-50:]),
            'max_rew': np.max(self.episode_rewards[-50:])
        }

        print('Recording episode {}'.format(len(self.episode_rewards)))
        [print('{} - {}'.format(k, v)) for k, v in summaries.items()]

        for tag, value in summaries.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=float(value))]
            )
            self.writer.add_summary(summary, len(self.episode_rewards))

        self.current_episode_rewards = []
