""" tools for analyzing results of experiments """
import os
from os.path import join

import pandas as pd
import numpy as np

from energy_py.common.utils import load_args

results_path = '/Users/adam/git/energy_py/energy_py/experiments/results/'


def load_env_args(run_name):
    return load_args(
        join(results_path, run_name, 'agent_args.txt')
    )


def load_agent_args(run_name):
    return load_args(
        join(results_path, run_name, 'agent_args.txt')
    )


def read_run_episodes(run_name, verbose=False):

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
    """ Processes a single episode episodeory - aka the info dict """

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


class Run(object):
    def __init__(
            self,
            expt_name,
            run_name
    ):
        self.run_name = run_name
        path = join(expt_name, self.run_name)

        self.agent_args = load_agent_args(path)
        self.env_args = load_env_args(path)

        self.episodes = read_run_episodes(
            path, verbose=False
        )

        self.output = process_run(self.episodes)

    def __call__(self):
        return self.output


if __name__ == '__main__':

    def process_experiment(expt_name, runs):
        """ Processes multiple runs into a single experiment summary """
        if isinstance(runs, str):
            runs = [runs]

        runs = {run_name: Run(expt_name, run_name)
                for run_name in runs}

        # delta creation could be a function TODO
        baseline = runs['no_op'].output['reward_per_day']

        for name, run in runs.items():
            run.output['delta_reward_per_day'] = run.output['reward_per_day'] - baseline


            print('{} {}'.format(
                run.run_name, run.output['delta_reward_per_day']
            ))

        return runs

    out = process_experiment('new_flex', runs=['no_op', 'autoflex'])

    #  create the no-nop baseline!!!
    #  don't need to do a run to do thisj
    #  but might be eaiser (can use same infrastructure as I just created)
    #  decided to do the second
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')


    def plot_time_series(
            data,
            y,
            figsize=(25, 10),
            fig_name=None,
            same_plot=False,
            **kwargs):

        if isinstance(y, str):
            y = [y]

        if same_plot:
            nrows = 1
        else:
            nrows = len(y)

        figsize = (100, 10)

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

    def plot_figures(plot_data):
        f = plot_time_series(
            plot_data,
            y=['reward', 'electricity_price'],
            fig_name='fig1.png'
        )

        f = plot_time_series(
            plot_data,
            y=['site_demand', 'site_electricity_consumption'],
            same_plot=True,
            fig_name='fig2.png'
        )

        f = plot_time_series(
            plot_data,
            y=['site_demand', 'electricity_price', 'setpoint', 'demand_delta'],
            fig_name='fig3.png'
        )

        f = plot_time_series(
            plot_data,
            y=['electricity_price', 'setpoint', 'charge'],
            fig_name='fig4.png'
        )

    plot_figures(out['autoflex'].episodes[0].iloc[:48, :])
    plot_figures(out['autoflex'].episodes[0].iloc[:, :])
