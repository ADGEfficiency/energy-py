"""
tools for analyzing results of experiments

energy_py experiments are structured

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
"""

import os
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_py.common.utils import load_args


plt.style.use('ggplot')

results_path = './results/'

def load_env_args(run_name):
    return load_args(
        join(results_path, run_name, 'agent_args.txt')
    )


def load_agent_args(run_name):
    return load_args(
        join(results_path, run_name, 'agent_args.txt')
    )


def read_run_episodes(run_name, verbose=False):
    """ Reads all episodes for a single run """

    episodes = []
    for root, dirs, files in os.walk(
            join(results_path, run_name, 'env_histories')
    ):

        for f in files:
            episodes.append(pd.read_csv(join(root, f), index_col=0, parse_dates=True))
    if verbose:
        print('read {} episodes'.format(len(episodes)))

    return episodes


def process_episode(episode, verbose=False):
    """ Process a single episode - aka the info dict returned by env.step() """

    #  these should go before __init__
    num_5mins_per_day = 12 * 24
    reward_per_5min = episode['reward'].sum() / episode.shape[0]

    summary = {
        'total_episode_reward': episode['reward'].sum(),
        'avg_electricity_price': episode['electricity_price'].mean(),
        'no_ops': episode[episode['action'] == 0].shape[0] / episode.shape[0],
        'count_increase_setpoint': episode[episode['action'] == 1].shape[0],
        'count_decrease_setpoint': episode[episode['action'] == 2].shape[0],

        'reward_per_5min': reward_per_5min,
        'reward_per_day': reward_per_5min * num_5mins_per_day
    }

    if verbose:
        [print('{} {:2.0f}'.format(k, v)) for k, v in summary.items()]

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
        'no_ops': run_summary['no_ops'].mean(),

        'reward_per_5min': run_summary['reward_per_5min'].mean(),
        'reward_per_day': run_summary['reward_per_day'].mean()
    }

    return run_summary


def process_experiment(expt_name, runs):
    """ Processes multiple runs into a single experiment summary """
    if isinstance(runs, str):
        runs = [runs]

    if 'no_op' not in runs:
        runs.append('no_op')

    runs = {run_name: Run(expt_name, run_name)
            for run_name in runs}

    # delta creation could be a function TODO
    baseline = runs['no_op'].summary['reward_per_day']

    for name, run in runs.items():
        run.summary['delta_reward_per_day'] = run.summary['reward_per_day'] - baseline

        if name is not 'no_op':
            print('{} reward per day vs no_op $/day {:2.2f}'.format(
                run.run_name, run.summary['delta_reward_per_day']
            ))

    return runs


def plot_time_series(
        data,
        y,
        figsize=[25, 10],
        fig_name=None,
        same_plot=False,
        **kwargs):

    if isinstance(y, str):
        y = [y]

    if same_plot:
        nrows = 1

    else:
        nrows = len(y)

    figsize[1] = 2 * nrows

    f, a = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    a = np.array(a).flatten()

    for idx, y_label in enumerate(y):
        if same_plot:
            idx = 0
        a[idx].set_title(y_label)
        data.plot(y=y_label, ax=a[idx], **kwargs)

    if fig_name:
        f.savefig(fig_name)

    return f


def plot_figures(plot_data, fig_path='./'):

    f = plot_time_series(
        plot_data,
        y=['site_demand', 'site_electricity_consumption'],
        same_plot=True,
        fig_name=join(fig_path, 'fig1.png')
    )

    f = plot_time_series(
        plot_data,
        y=['electricity_price', 'setpoint', 'delta_demand', 'charge'],
        fig_name=join(fig_path, 'fig2.png')
    )


class Run(object):
    def __init__(
            self,
            expt_name,
            run_name
    ):
        self.expt_name = expt_name
        self.run_name = run_name
        path = join(expt_name, self.run_name)

        self.agent_args = load_agent_args(path)
        self.env_args = load_env_args(path)

        self.episodes = read_run_episodes(
            path, verbose=False
        )

        self.summary = process_run(self.episodes)

    def __call__(self):
        return self.summary


if __name__ == '__main__':

    runs = process_experiment(
        'new_flex',
        ['autoflex', 'random', 'no_op']
    )

    autoflex = runs['autoflex']

    last_ep = autoflex.episodes[-1]


    plot_figures(last_ep.iloc[-288:, :],
                 fig_path=results_path)

    def run_markdown_writer(
            run,
            path
    ):
        with open(join(path, run.run_name, 'run_results.md'), 'w') as text_file:
                text_file.write(
                    '## {} run of the {} experiment'.format(
                        run.run_name, run.expt_name) + os.linesep)

                text_file.write(
                    '### delta versus the no_op case' + os.linesep)

                text_file.write(
                    '$/day {:2.2f}'.format(
                        run.summary['delta_reward_per_day']) + os.linesep)

                text_file.write(
                    '$/yr {:2.0f}'.format(
                        run.summary['delta_reward_per_day'] * 365) + os.linesep)

                text_file.write('![img](fig1.png)' + os.linesep)

    run_markdown_writer(autoflex, results_path)

    def expt_markdown_writer(
            runs,
            path
    ):
        with open(join(path, 'expt_results.md'), 'w') as text_file:

            for run_name, run in runs.items():
                text_file.write('## ' + run_name + os.linesep)

                text_file.write(
                    '$/day {:2.2f}'.format(
                        run.summary['delta_reward_per_day']) + os.linesep)

                text_file.write(
                    '$/yr {:2.0f}'.format(
                        run.summary['delta_reward_per_day'] * 365) + os.linesep)

    expt_markdown_writer(runs, results_path)
