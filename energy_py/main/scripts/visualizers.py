"""
"""

import collections
import os

import matplotlib.pyplot as plt
import pandas as pd

from energy_py.main.scripts.utils import ensure_dir


class Visualizer(object):
    """
    A base class to create charts.

    Args:

    """
    def __init__(self):
        self.base_path = None
        self.outputs   = collections.defaultdict(list)

    def _output_results(self, action): raise NotImplementedError

    def output_results(self):
        """
        The main visualizer function

        Purpose is to output results from the object

        Should be overridden in the most child class - usually an env
        specific Visualizer object

        Envionsed this will be called in the env.output_results() fctn
        """
        return self._output_results()

    def make_time_series_fig(self, df, cols, xlabel, ylabel):
        """
        makes a time series figure from a dataframe and specified columns
        """
        #  make the figure & axes objects
        fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        for col in cols:
            data = df.loc[:, col].astype(float)
            data.plot(kind='line', ax=ax, label=col)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        return fig


class Env_Episode_Visualizer(Visualizer):
    """
    A base class to create charts for a single episode_length

    Args:
        env_info (dictionary) : episode results
        state_ts (pandas df)  : time series object for the environment
        episode  (int)        : the episode number
    """

    def __init__(self, env_info, state_ts, episode):
        super().__init__()

        self.env_info = env_info
        self.state_ts = state_ts
        self.episode = episode
        self.base_path = os.path.join('results/episodes/')

    def make_dataframe(self):
        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.env_info)
        # TODO get the index working properly
        # self.outputs['dataframe'].index = self.state_ts.index[:len(self.outputs['dataframe'])]

        path = os.path.join(self.base_path, 'env_history_{}.csv'.format(self.episode))
        ensure_dir(path)
        self.outputs['dataframe'].to_csv(path)

        return self.outputs['dataframe']

    def print_results(self):
        RL_cost = sum(self.env_info['RL_cost'])
        BAU_cost = sum(self.env_info['BAU_cost'])
        steps = self.outputs['dataframe'].loc[:, 'steps']
        steps = steps.iloc[-1]

        print('Episode {} ran for {} steps'.format(self.episode, steps))
        print('RL cost was {}'.format(RL_cost))
        print('BAU cost was {}'.format(BAU_cost))
        print('Savings were {}'.format(BAU_cost-RL_cost))

        hours_per_step = 5/60
        episode_run_time = hours_per_step * steps
        daily_avg_saving = episode_run_time * 24
        # print('Daily average saving was {}'.format(daily_avg_saving))

        return None

    def make_electricity_cost_fig(self):
        fig = self.make_time_series_fig(self.outputs['dataframe'],
                                                          cols=['BAU_cost',
                                                                'RL_cost',
                                                                'electricity_price'],
                                                          ylabel='Cost to deliver electricity [$/hh]',
                                                          xlabel='Time')

        path = os.path.join(self.base_path, 'electricity_cost_fig_{}.png'.format(self.episode))
        ensure_dir(path)
        fig.savefig(path)
        return fig


class Agent_Memory_Visualizer(Visualizer):
    """
    A class to create graphs from agent memory.
    """
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join('results/')
        pass
