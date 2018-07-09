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


def process_episode(hist, verbose=False):
    """ Processes a single episode history - aka the info dict """

    summary = {
        'total_episode_reward': hist['reward'].sum(),
        'avg_electricity_price': hist['electricity_price'].mean(),
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

    #  these should go before __init__
    # num_5mins_per_day = 12 * 24
    # num_5mins_per_year = num_5mins_per_day * 365

    avg_ep_reward = run_summary['total_episode_reward'].mean()
    run_summary = {
        'avg_ep_reward': avg_ep_reward,
        'number_episodes': len(episodes),
        'number_loss_episodes': run_summary[run_summary['total_episode_reward'] < 0].shape[0]
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

        episodes = read_run_episodes(
            'new_flex/autoflex', verbose=True
        )

        self.output = process_run(episodes)

    def __call__(self):
        return self.output


if __name__ == '__main__':

    def process_experiment(expt_name, runs):
        """ Processes multiple runs into a single experiment summary """
        if isinstance(runs, str):
            runs = [runs]

        runs = {run_name: Run(expt_name, run_name)
                for run_name in runs}

        baseline = runs['no_op'].output['avg_ep_reward']

        for name, run in runs.items():
            run.output['baseline_delta'] = run.output['avg_ep_reward'] - baseline

            print('{} {}'.format(run.run_name, run.output))

        return runs

    out = process_experiment('new_flex', runs=['no_op', 'autoflex'])

    #  create the no-nop baseline!!!
    #  don't need to do a run to do thisj
    #  but might be eaiser (can use same infrastructure as I just created)
    #  decided to do the second



