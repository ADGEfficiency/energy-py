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

    def make_time_series_fig(self, df, cols, xlabel, ylabel, ylim=[]):
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

        if ylim:
            ax.set_ylim(ylim)
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

        saving = BAU_cost-RL_cost

        print('Episode {} ran for {} steps'.format(self.episode, steps))
        print('RL cost was {}'.format(RL_cost))
        print('BAU cost was {}'.format(BAU_cost))
        print('Savings were {}'.format(saving))

        #  TODO should take env time step into account
        #  here is hardcoded as 5 minutes
        avg_saving_per_hour = saving / (steps / 12)
        avg_saving_per_day = 24 * avg_saving_per_hour

        print('Mean hourly saving was {}'.format(avg_saving_per_hour))
        print('Mean daily saving was {}'.format(avg_saving_per_day))
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
    A class to create outputs from agent memory.
    """
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join('results/')

    def make_dataframes(self):
        """
        Helper function for self.output_results()

        Creates two dataframes
        'dataframe_steps'    = dataframe on a step frequency
        'dataframe_episodic' = dataframe on a episodic frequency
        """
        #  create lists on a step by step basis
        episodes = [exp.episode for exp in self.experiences]
        steps = [exp.step for exp in self.experiences]
        observations = [exp.observation for exp in self.experiences]
        actions = [exp.action for exp in self.experiences]
        rewards = [exp.reward for exp in self.experiences]
        scaled_rewards = [exp.reward for exp in self.scaled_experiences]
        discounted_returns =  [exp.discounted_return for exp in self.scaled_experiences]

        df_dict = {
                   'episode':episodes,
                   'step':steps,
                   'observation':observations,
                   'action':actions,
                   'reward':rewards,
                   'scaled_reward':scaled_rewards,
                   'discounted_return':discounted_returns
                   }

        dataframe_steps = pd.DataFrame.from_dict(df_dict)

        #  expanding out the tuples
        for col in dataframe_steps.columns:
            pd.DataFrame(dataframe_steps.loc[: ,col].values.tolist(),
                         index=dataframe_steps.index)

        dataframe_episodic = dataframe_steps.groupby(by=['episode'],
                                                  axis=0).sum()

        if self.losses:
            dataframe_episodic.loc[:, 'loss'] = self.losses

        dataframe_steps.set_index('episode', drop=True, inplace=True)
        # dataframe_episodic.set_index('episode', drop=True, inplace=True)

        path_steps = os.path.join(self.base_path, 'agent_df_steps.csv')
        ensure_dir(path_steps)
        dataframe_steps.to_csv(path_steps)

        path_episodic = os.path.join(self.base_path, 'agent_df_episodic.csv')
        ensure_dir(path_episodic)
        dataframe_episodic.to_csv(path_episodic)
        return dataframe_steps,  dataframe_episodic

    def make_returns_fig(self):
        """
        Makes a plot of undiscounted reward per episode
        """
        fig = self.make_time_series_fig(self.outputs['dataframe_episodic'],
                                                          cols=['reward'],
                                                          ylabel='Undiscounted total reward per episode',
                                                          xlabel='Episode')

        path = os.path.join(self.base_path, 'return_per_episode.png')
        ensure_dir(path)
        fig.savefig(path)
        return fig

    def make_loss_fig(self):
        """
        Makes a plot of undiscounted reward per episode
        """
        fig = self.make_time_series_fig(self.outputs['dataframe_episodic'],
                                                          cols=['loss'],
                                                          ylabel='Loss',
                                                          xlabel='Episode')

        path = os.path.join(self.base_path, 'loss_per_episode.png')
        ensure_dir(path)
        fig.savefig(path)
        return fig

    def _output_results(self):
        """
        The main function to output results from the memory

        Overridden method of the Visualizer parent class

        Envionsed to run after all episodes are finished
        """
        #  making the two summary dataframes
        self.outputs['dataframe_steps'], self.outputs['dataframe_episodic'] = self.make_dataframes()

        self.outputs['returns_fig'] = self.make_returns_fig()

        if self.losses:
            self.outputs['loss_fig'] = self.make_loss_fig()
        return self.outputs
