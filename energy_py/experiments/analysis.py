"""
TODO - clean up how the paths flow through Run

tools for analyzing results of experiments

the code takes advantage of the structure in an energy_py experiment

episodes - runs - experiments

A single run has multiple episodes
A single experiment has multiple runs

experiment_1
    run_1
        episode_1, episode_2 ... episode_n
    run_2
        episode_1, episode_2 ... episode_n
    ...

experiment_2
    run_1
        episode_1, episode_2 ... episode_n
    run_2
        episode_1, episode_2 ... episode_n
    ...

The code is organized by

- process_episode/run/experiment (3 funcs)
- plot_episode/run/experiment (3 funcs)

"""
import os
from os.path import join

import numpy as np
import pandas as pd

from energy_py.common.utils import load_args
from energy_py.experiments.markdown_writers import expt_markdown_writer
from energy_py.experiments.plotting import plot_flex_episode, plot_time_series, plot_battery_episode


results_path = './results/'


def read_run_episodes(path):
    """ Reads all episodes for a single run """

    episodes = []
    for root, dirs, files in os.walk(
            join(path, 'env_histories')
    ):

        for f in files:
            episodes.append(pd.read_csv(join(root, f), index_col=0, parse_dates=True))

    print('{} {} episodes'.format(path, len(episodes)))

    return episodes


def process_episode(episode):
    """ Process a single episode - aka the info dict returned by env.step() """

    #  these should go before __init__
    num_5mins_per_day = 12 * 24
    reward_per_5min = episode['reward'].sum() / episode.shape[0]

    summary = {
        'total_reward': episode['reward'].sum(),
        # 'no_ops': episode[episode['action'] == 0].shape[0] / episode.shape[0],
        # 'count_increase_setpoint': episode[episode['action'] == 1].shape[0],
        # 'count_decrease_setpoint': episode[episode['action'] == 2].shape[0],

        # 'reward_per_5min': reward_per_5min,
        # 'reward_per_day': reward_per_5min * num_5mins_per_day
    }

    return summary


def process_run(episodes):
    """ Processes multiple episodes into a summary for a single run """

    for episode in episodes:
        episode_summaries = process_episode(episode)

    run_summary = pd.DataFrame(
        episode_summaries,
        index=np.arange(1, len(episodes) + 1)
    )

    run_summary.index.rename('episode')

    avg_ep_reward = run_summary['total_episode_reward'].mean()

    run_summary = {
        'avg_ep_reward': avg_ep_reward,
        'num_episodes': len(episodes),
        'num_loss_episodes': run_summary[run_summary['total_episode_reward'] < 0].shape[0],
        # 'no_ops': run_summary['no_ops'].mean(),

        # 'reward_per_5min': run_summary['reward_per_5min'].mean(),
        # 'reward_per_day': run_summary['reward_per_day'].mean()
    }

    return run_summary


def process_experiment(expt_name, runs):
    """ Processes multiple runs into a single experiment summary """
    if isinstance(runs, str):
        runs = [runs]

    print('Processing expt {} runs {}'.format(expt_name, runs))

    runs = {run_name: Run(expt_name, run_name)
            for run_name in runs}

    plot_experiment(runs, expt_name)

    expt_markdown_writer(runs, join(results_path, expt_name))

    return runs


class Run(object):
    def __init__(
            self,
            expt_name,
            run_name
    ):
        print('Processing run {}'.format(run_name))
        self.expt = expt_name
        self.name = run_name
        path = join(results_path, self.expt, self.name)

        #  args are dicts
        self.agent_args = load_args(
            join(path, 'agent_args.txt')
        )
        self.env_args = load_args(
            join(path, 'env_args.txt')
        )

        #  self.episodes is a list TODO
        # self.episodes = read_run_episodes(path)
        self.episode_rewards = pd.read_csv(
            join(path, 'episode_rewards.csv'),
            index_col=0
        )

        self.episode_rewards.columns = [self.name]

        #  summary is a dict
        # self.summary = process_run(self.episodes)

        if self.env_args['env_id'] == 'flex':
            plot_ep = 5
            for episode in range(len(self.episodes))[-plot_ep:]:
                print('plotting last {} episodes'.format(plot_ep))

                plot_flex_episode(
                    self.episodes[episode].iloc[-288:, :],
                    fig_path=join(
                        results_path,
                        self.expt,
                        elf.name,
                        'episode_{}'.format(episode)
                    )
                )

        if self.env_args['env_id'] == 'battery':
            plot_ep = 5
            for episode in range(len(self.episodes))[-plot_ep:]:
                print('plotting last {} episodes'.format(plot_ep))

                plot_battery_episode(
                    self.episodes[episode].iloc[-288:, :],
                    fig_path=join(
                        results_path,
                        self.expt,
                        elf.name,
                        'episode_{}'.format(episode)
                    )
                )

    def __call__(self):
        return self.summary


def plot_run(run):

    plot_time_series(
        run.episode_rewards,
        'total_reward',
        kind='line',
        fig_path=join(
            results_path,
            run.expt,
            run.name,
            'total_episode_rewards.png'
        )
    )


def plot_experiment(runs, expt_name):

    all_eps = pd.concat(
        [run.episode_rewards for _, run in runs.items()],
        axis=1
    )

    plot_time_series(
        all_eps,
        all_eps.columns,
        same_plot=True,
        fig_path=join(
            results_path,
            expt_name,
            'total_episode_rewards.png'
        )
    )


if __name__ == '__main__':

    runs = process_experiment(
        'cartpole_example',
        ['dqn', 'random']
    )
