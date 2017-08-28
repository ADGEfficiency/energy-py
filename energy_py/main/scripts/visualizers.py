"""
"""

import collections
import itertools
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

    def make_figure(self,
                    df,
                    cols,
                    xlabel,
                    ylabel,
                    xlim='all',
                    ylim=[],
                    path=None):

        fig = self.make_time_series_fig(df, cols, xlabel, ylabel, xlim, ylim)
        if path:
            self.save_fig(fig, path)
        return fig

    def make_time_series_fig(self, df, cols, xlabel, ylabel,  xlim='all', ylim=[]):
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

        if xlim == 'last_week':
            start = df.index[-7 * 24 * 12]
            end = df.index[-1]

        if xlim == 'last_month':
            start = df.index[-30 * 24 * 12]
            end = df.index[-1]

        if xlim == 'all':
            start = df.index[0]
            end = df.index[-1]

        ax.set_xlim([start, end])

        return fig

    def save_fig(self, fig, path):
        ensure_dir(path)
        fig.savefig(path)


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
        #  create the output dataframe
        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.env_info)
        return self.outputs['dataframe']

    def print_results(self):
        """
        not sure if this should be here!
        """
        RL_cost = sum(self.env_info['RL_cost_[$/5min]'])
        BAU_cost = sum(self.env_info['BAU_cost_[$/5min]'])
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
        print('agent memory is making dataframes')
        assert len(self.experiences) == len(self.scaled_experiences)

        ep, stp, obs, act, rew, nxt_obs = [], [], [], [], [], []
        scl_obs, scl_act, scl_rew, scl_nxt_obs, dis_ret = [], [], [], [], []
        for exp, scaled_exp in itertools.zip_longest(self.experiences, self.scaled_experiences):
            ep.append(exp.episode)
            stp.append(exp.step)
            obs.append(exp.observation)
            act.append(exp.action)
            rew.append(exp.reward)
            nxt_obs.append(exp.next_observation)

            scl_obs.append(scaled_exp.observation)
            scl_act.append(scaled_exp.action)
            scl_rew.append(scaled_exp.reward)
            scl_nxt_obs.append(scaled_exp.next_observation)
            dis_ret.append(scaled_exp.discounted_return)

        df_dict = {
                   'episode':ep,
                   'step':stp,
                   'observation':obs,
                   'action':act,
                   'reward':rew,
                   'next_observation':nxt_obs,
                   'scaled_reward':scl_rew,
                   'discounted_return':dis_ret,
                   'scaled_observation':scl_obs,
                   'scaled_action':scl_act,
                   'scaled_reward':scl_rew,
                   'scaled_next_observation':scl_nxt_obs,
                   }

        dataframe_steps = pd.DataFrame.from_dict(df_dict)

        dataframe_episodic = dataframe_steps.groupby(by=['episode'],
                                                  axis=0).sum()

        if self.losses:
            dataframe_episodic.loc[:, 'loss'] = self.losses

        dataframe_steps.set_index('episode', drop=True, inplace=True)

        return dataframe_steps,  dataframe_episodic

    def _output_results(self):
        """
        The main function to output results from the memory

        Overridden method of the Visualizer parent class

        Envionsed to run after all episodes are finished
        """
        #  making the two summary dataframes
        self.outputs['dataframe_steps'], self.outputs['dataframe_episodic'] = self.make_dataframes()
        return self.outputs

class Eternity_Visualizer(Visualizer):
    """
    A class to join together data generated by the agent and environment
    """
    def __init__(self, episode,
                       agent,
                       env):
        super().__init__()

        self.env = env
        self.agent = agent
        self.episode = episode

        self.base_path_agent = os.path.join('results/')
        self.base_path_env = os.path.join('results/episodes')

        #  pull out the data
        print('Eternity visualizer is pulling data out of the agent')
        self.agent_memory = self.agent.memory.output_results()
        print('Eternity visualizer is pulling data out of the environment')
        self.env_info = self.env.output_results()

        self.figures = collections.defaultdict(list)
        self.env_figures = self.env.episode_visualizer.figures

        self.state_ts = self.env.state_ts
        self.observation_ts = self.env.observation_ts

        #  use the index from the state_ts for the other len(total_steps) dataframes
        index = pd.to_datetime(self.state_ts.index)
        dfs = [self.env_info['dataframe']]

        for df in dfs:
            print(len(index))
            print(len(df))
            df.index = index


    def _output_results(self):
        """
        Generates results
        """
        def save_df(df, path):
            ensure_dir(path)
            df.to_csv(path)

        print('saving the figures')

        #  iterate over the figure dictionary from the env visualizer
        #  this exists to allow env specific figures to be collected by
        #  the Eternity_Visualizer

        #  TODO similar thing for agent!
        # for make_fig_fctn, fig_name in self.env_figures.items():
        #     self.figures[fig_name] = make_fig_fctn(self.env_info,
        #                                            self.base_path_env)

        self.figures['elect_cost'] = self.make_figure(df=self.env_info['dataframe'],
                                                      cols=['BAU_cost_[$/5min]',
                                                            'RL_cost_[$/5min]',
                                                            'electricity_price'],
                                                      ylabel='Cost to deliver electricity [$/5min]',
                                                      xlabel='Time',
                                                      xlim='all',
                                                      path=os.path.join(self.base_path_env,'electricity_cost_fig_{}.png'.format(self.episode)))

        self.figures['returns'] = self.make_figure(df=self.agent_memory['dataframe_episodic'],
                                                      cols=['reward'],
                                                      ylabel='Undiscounted total reward per episode',
                                                      xlabel='Episode',
                                                      path=os.path.join(self.base_path_agent, 'return_per_episode.png'))

        if self.agent.memory.losses:
            self.figures['loss'] = self.make_figure(df=self.agent_memory['dataframe_episodic'],
                                                          cols=['loss'],
                                                          ylabel='Loss',
                                                          xlabel='Episode',
                                                          path=os.path.join(self.base_path_agent, 'loss_per_episode.png'))

        print('saving env dataframe')
        save_df(self.env_info['dataframe'],
                os.path.join(self.base_path_env, 'env_history_{}.csv'.format(self.episode)))

        print('saving state dataframe')
        save_df(self.state_ts,
                os.path.join(self.base_path_env, 'state_ts_{}.csv'.format(self.episode)))

        print('saving memory steps dataframe')
        save_df(self.agent_memory['dataframe_steps'],
                os.path.join(self.base_path_agent, 'agent_df_steps.csv'))

        print('saving memory episodic dataframe')
        save_df(self.agent_memory['dataframe_episodic'],
                os.path.join(self.base_path_agent, 'agent_df_episodic.csv'))
