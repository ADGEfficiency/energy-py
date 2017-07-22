import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

import environments.base_env
import environments.library
import matplotlib.pyplot as plt


class env(environments.base_env.base_class):

    def __init__(self, episode_length, lag, random_ts, verbose):
        self.episode_length = episode_length
        self.lag = lag
        self.random_ts = random_ts
        self.verbose = verbose

        self.actual_state, self.visible_state = self.load_data(self.episode_length, self.lag, self.random_ts)
        self.state_models = [{'Name': 'Settlement period', 'Min': 0, 'Max': 48},
                             {'Name': 'HGH demand', 'Min': 0, 'Max': 30},
                             {'Name': 'LGH demand', 'Min': 0, 'Max': 20},
                             {'Name': 'Cooling demand', 'Min': 0, 'Max': 10},
                             {'Name': 'Electrical demand', 'Min': 0, 'Max': 20},
                             {'Name': 'Ambient temperature', 'Min': 0, 'Max': 30},
                             {'Name': 'Gas price', 'Min': 15, 'Max': 25},
                             {'Name': 'Import electricity price', 'Min': -200, 'Max': 1600},
                             {'Name': 'Export electricity price', 'Min': -200, 'Max': 1600}]

        self.asset_models = [
            environments.library.gas_engine(size=25, name='GT 1'),
            environments.library.gas_engine(size=25, name='GT 2')]

        self.state = self.reset()


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _step(self, actions):
        actual_state = self.actual_state.iloc[self.steps, 1:]
        time_stamp = pd.to_datetime(self.actual_state.iloc[self.steps, 0])

        if self.verbose > 0:
            self.asset_states()
            print(actions)

        # taking actions
        for k, asset in enumerate(self.asset_models):
            for var in asset.variables:
                action = actions[k]

                # case at full load
                if var['Current'] == var['Max']:
                    var['Current'] = min([var['Max'],
                                                 var['Current'] + action])

                # case at minimum load
                elif var['Current'] == var['Min']:
                    if (var['Current'] + action) < var['Min']:
                        var['Current'] = 0
                    elif (var['Current'] + action) >= var['Min']:
                        var['Current'] = var['Min'] + action

                # case at off
                elif var['Current'] == 0:
                    if action < 0:
                        var['Current'] = 0
                    else:
                        var['Current'] = var['Min']

                # case in all other times
                else:
                    new = var['Current'] + action
                    new = min(new, var['Max'])
                    new = max(new, var['Min'])
                    var['Current'] = new

                asset.update()
        if self.verbose > 0:
            self.asset_states()
        # sum of energy inputs/outputs for all assets
        total_gas_burned = sum([asset.gas_burnt for asset in self.asset_models])
        total_HGH_gen = sum([asset.HG_heat_output for asset in self.asset_models])
        total_LGH_gen = sum([asset.LG_heat_output for asset in self.asset_models])
        total_COOL_gen = sum([asset.cooling_output for asset in self.asset_models])
        total_elect_gen = sum([asset.power_output for asset in self.asset_models])

        # energy demands
        elect_dem = actual_state['Electrical']
        HGH_dem = actual_state['HGH']
        LGH_dem = actual_state['LGH']
        COOL_dem = actual_state['Cooling']

        # energy balances
        HGH_bal = HGH_dem - total_HGH_gen
        LGH_bal = LGH_dem - total_LGH_gen
        COOL_bal = COOL_dem - total_COOL_gen

        # backup gas boiler to pick up excess load
        backup_blr = max(0, HGH_bal) + max(0, LGH_bal)
        gas_burned = total_gas_burned + (backup_blr / 0.8)

        # backup electric chiller for cooling load
        backup_chiller = max(0, COOL_bal)
        backup_chiller_elect = backup_chiller / 3
        elect_dem += backup_chiller_elect

        # electricity balance
        elect_bal = elect_dem - total_elect_gen
        import_elect = max(0, elect_bal)
        export_elect = abs(min(0, elect_bal))

        # all prices in £/MWh
        gas_price = actual_state['Gas price']
        import_price = actual_state['Import electricity price']
        export_price = actual_state['Export electricity price']
        gas_cost = (gas_price * gas_burned) / 2  # £/HH
        import_cost = (import_price * import_elect) / 2  # £/HH
        export_revenue = (export_price * export_elect) / 2  # £/HH

        reward = export_revenue - (gas_cost + import_cost)  # £/HH

        SP = actual_state['Settlement period']
        total_heat_demand = HGH_dem + LGH_dem
        self.info.append([SP,
                          total_elect_gen,
                          import_price,
                          total_heat_demand,
                          time_stamp])

        self.steps += int(1)
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True

        next_state = self.visible_state.iloc[self.steps, 1:] # visible state
        self.state = next_state

        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.action_space = self.create_action_space()

        return next_state, reward, self.done, self.info

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _load_data(self, episode_length, lag, random_ts):
        ts = pd.read_csv('environments/time_series.csv', index_col=[0])
        ts.iloc[:, 1:] = ts.iloc[:, 1:].apply(pd.to_numeric)
        ts.loc[:, 'Timestamp'] = ts.loc[:, 'Timestamp'].apply(pd.to_datetime)

        if random_ts:
            idx = random.randint(0, self.episode_length)

        ts = ts.iloc[idx:idx+self.episode_length, :]

        if lag < 0:
            actual_state = ts.iloc[:lag, :]
            visible_state = ts.shift(lag).iloc[:lag, :]

        elif lag == 0:
            actual_state = ts.iloc[:, :]
            visible_state = ts.iloc[:, :]

        elif lag > 0:
            actual_state = ts.iloc[lag:, :]
            visible_state = ts.shift(lag).iloc[lag:, :]

        assert actual_state.shape == visible_state.shape

        return visible_state, actual_state

    def _create_action_space(self):
        action_space = []
        for j, asset in enumerate(self.asset_models):
            radius = asset.variables[0]['Radius']
            space = gym.spaces.Box(low=-radius,
                                   high=radius,
                                   shape=(1))
            action_space.append(space)
        return action_space

    def _make_outputs(self, path):
        def fig_engineering(env_info):
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8*2, 6*2)
            print(env_info.shape)
            print(env_info.index)
            env_info.plot(y=['Power generated [MWe]',
                             'Import electricity price [£/MWh]',
                             'Total heat demand [MW]'],
                              subplots=False,
                              kind='line',
                              use_index=True,
                              ax=ax)

            ax.legend(loc='best', fontsize=18)
            ax.set_xlabel('Steps (Last Episode)')
            ax.set_title('Operation for Last Episode')
            fig.savefig(path+'figures/fig_engineering.png')
            return fig

        env_info = pd.DataFrame(self.info,
                                columns=['Settlement period',
                                         'Power generated [MWe]',
                                         'Import electricity price [£/MWh]',
                                         'Total heat demand [MW]',
                                         'Timestamp'])
        env_info.loc[:, 'Timestamp'] = env_info.loc[:, 'Timestamp'].apply(pd.to_datetime)
        env_info.set_index(keys='Timestamp', inplace=True, drop=True)
        f1 = fig_engineering(env_info)
        return env_info
