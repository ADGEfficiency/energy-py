import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding

import environments.base_env
import environments.library


class env(environments.base_env.base_class):

    def __init__(self, episode_length, lag, verbose):
        self.episode_length = episode_length
        self.lag = lag
        self.verbose = verbose

        self.ts = self.load_data(self.episode_length)
        self.state_models = [
            {'Name': 'Settlement period', 'Min': 0, 'Max': 48},
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
            environments.library.gas_engine(size=25, name='GT 2'),
            environments.library.gas_engine(size=25, name='GT 3')]

        self.state = self.reset()


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _step(self, actions):
        # true state is ahead of the visible state
        # true state is totally hidden from agent
        true_state = self.ts.iloc[self.steps + self.lag, 1:]

        # taking actions
        for k, asset in enumerate(self.asset_models):
            for var in asset.variables:
                action = actions[k]

                if var['Current'] == 0 and action > 0:
                    # if off before and we do any positive action we turn on
                    var['Current'] = var['Min']

                elif (var['Current'] + action) < var['Min'] and var['Current'] > var['Min']:
                    # if we decrease load below the var['Min'] we turn off
                    var['Current'] = var['Min']

                elif (var['Current'] + action) < var['Min'] and var['Current'] == var['Min']:
                    var['Current'] = 0

                elif (var['Current'] + action) >= var['Max']:
                    var['Current'] = var['Max']

                else:
                    var['Current'] = var['Current'] + action

            asset.update()

        # sum of energy inputs/outputs for all assets
        total_gas_burned = sum([asset.gas_burnt for asset in self.asset_models])
        total_HGH_gen = sum([asset.HG_heat_output for asset in self.asset_models])
        total_LGH_gen = sum([asset.LG_heat_output for asset in self.asset_models])
        total_COOL_gen = sum([asset.cooling_output for asset in self.asset_models])
        total_elect_gen = sum([asset.power_output for asset in self.asset_models])

        # energy demands
        elect_dem = true_state['Electrical']
        HGH_dem = true_state['HGH']
        LGH_dem = true_state['LGH']
        COOL_dem = true_state['Cooling']

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
        gas_price = true_state['Gas price']
        import_price = true_state['Import electricity price']
        export_price = true_state['Export electricity price']
        gas_cost = (gas_price * gas_burned) / 2  # £/HH
        import_cost = (import_price * import_elect) / 2  # £/HH
        export_revenue = (export_price * export_elect) / 2  # £/HH

        reward = export_revenue - (gas_cost + import_cost)  # £/HH

        SP = true_state['Settlement period']
        total_heat_demand = HGH_dem + LGH_dem
        self.info.append([SP,
                          total_elect_gen,
                          import_price,
                          total_heat_demand])

        self.steps += int(1)
        if self.steps == (len(self.ts) - self.lag - 1):  #TODO do I need 1 here?
            self.done = True

        next_state = self.ts.iloc[self.steps, 1:].values # visible state
        self.state = next_state # visible state

        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.action_space = self.create_action_space()

        return next_state, reward, self.done, self.info

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _load_data(self, episode_length):
        ts = pd.read_csv('environments/time_series.csv', index_col=[0])
        ts = ts.iloc[:episode_length, :]
        ts.iloc[:, 1:] = ts.iloc[:, 1:].apply(pd.to_numeric)
        ts.loc[:, 'Timestamp'] = ts.loc[:, 'Timestamp'].apply(pd.to_datetime)
        return ts

    def _create_action_space(self):
        action_space = []
        for j, asset in enumerate(self.asset_models):
            radius = asset.variables[0]['Radius']
            space = gym.spaces.Box(low=-radius,
                                   high=radius,
                                   shape=(1))
            action_space.append(space)
        return action_space
    #
    # def _create_action_space_OLD(self, last_actions):
    #     # available actions are not constant - depend on asset current var
    #     # spaces = used to define legitimate action space
    #     actions, lows, highs = [], [], []
    #     for j, asset in enumerate(self.asset_models):
    #         current = last_actions[j]
    #         radius = asset.variables[0]['Radius']
    #         current_min = asset.variables[0]['Min']
    #         current_max = asset.variables[0]['Max']
    #         lower_bound = max(current - radius, current_min)
    #         upper_bound = min(current + radius, current_max)
    #
    #         off = gym.spaces.Box(low=0,
    #                              high=0,
    #                              shape=(1))
    #
    #         minimum = gym.spaces.Box(low=current_min,
    #                                  high=current_min,
    #                                  shape=(1))
    #
    #         current_space = gym.spaces.Box(low=lower_bound,
    #                                        high=upper_bound,
    #                                        shape=(1))
    #
    #         if current == 0:  # off
    #             action = gym.spaces.Tuple((off, minimum))
    #             low = min(off.low, minimum.low)
    #             high = max(off.high, minimum.high)
    #         elif current == current_min:  # at minimum load
    #             action = gym.spaces.Tuple((off, current_space))
    #             low = min(off.low, current_space.low)
    #             high = max(off.high, current_space.high)
    #         else:
    #             action = current_space
    #             low = current_space.low
    #             high = current_space.high
    #         actions.append(action)
    #         lows.append(low)
    #         highs.append(high)
    #     return actions, lows, highs
