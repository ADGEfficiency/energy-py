import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding

import assets.library


class energy_py(gym.Env):

    def __init__(self, episode_length):
        self.verbose = 0
        self.ts = self.load_data(episode_length)
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
            assets.library.gas_engine(size=20, name='GT 1')]

        self.state_names = [d['Name'] for d in self.state_models]
        self.action_names = [var['Name']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.s_mins, self.s_maxs = self.state_mins_maxs()
        self.a_mins, self.a_maxs = self.asset_mins_maxs()
        self.mins = np.append(self.s_mins, self.a_mins)
        self.maxs = np.append(self.s_maxs, self.a_maxs)

        self.seed()
        self.reset()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _seed(self, seed=None):  # taken straight from cartpole
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.steps = int(0)
        self.state = self.ts.iloc[0, 1:].values
        self.info = []
        self.done = False
        [asset.reset for asset in self.asset_models]
        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.observation_space = self.create_obs_space()
        self.action_space = self.create_action_space(self.last_actions)
        return self.state

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # take actions
        count = 0
        for asset in self.asset_models:
            for var in asset.variables:
                var['Current'] = action[count]
                count += 1
            asset.update()

        # go to next state
        # actions are applied to the next state
        # unseen by the agent when taking action - intended behaviour
        self.steps += int(1)

        state_ = self.ts.iloc[self.steps, 1:]
        self.state = state_.values

        # sum of energy inputs/outputs for all assets
        total_gas_burned = sum([asset.gas_burnt for asset in self.asset_models])
        total_HGH_gen = sum([asset.HG_heat_output for asset in self.asset_models])
        total_LGH_gen = sum([asset.LG_heat_output for asset in self.asset_models])
        total_COOL_gen = sum([asset.cooling_output for asset in self.asset_models])
        total_elect_gen = sum([asset.power_output for asset in self.asset_models])

        # energy demands
        elect_dem = state_['Electrical']
        HGH_dem = state_['HGH']
        LGH_dem = state_['LGH']
        COOL_dem = state_['Cooling']

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
        gas_price = state_['Gas price']
        import_price = state_['Import electricity price']
        export_price = state_['Export electricity price']
        gas_cost = (gas_price * gas_burned) / 2  # £/HH
        import_cost = (import_price * import_elect) / 2  # £/HH
        export_revenue = (export_price * export_elect) / 2  # £/HH

        reward = export_revenue - (gas_cost + import_cost)  # £/HH

        if self.steps == len(self.ts) - 1:
            self.done = True

        next_state = self.state
        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.action_space = self.create_action_space(self.last_actions)

        SP = state_['Settlement period']
        total_heat_demand = HGH_dem + LGH_dem
        self.info.append([SP,
                          total_elect_gen,
                          import_price,
                          total_heat_demand]
                         )

        return next_state, reward, self.done, self.info

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def load_data(self, episode_length):
        ts = pd.read_csv('assets/time_series.csv', index_col=[0])
        ts = ts.iloc[:episode_length, :]
        ts.iloc[:, 1:] = ts.iloc[:, 1:].apply(pd.to_numeric)
        ts.loc[:, 'Timestamp'] = ts.loc[:, 'Timestamp'].apply(pd.to_datetime)
        return ts

    def create_obs_space(self):
        states, self.state_names = [], []
        for mdl in self.state_models:
            states.append([mdl['Min'], mdl['Max']])
            self.state_names.append(mdl['Name'])
        return spaces.MultiDiscrete(states)

    def state_mins_maxs(self):
        s_mins, s_maxs = np.array([]), np.array([])
        for mdl in self.state_models:
            s_mins = np.append(s_mins, mdl['Min'])
            s_maxs = np.append(s_maxs, mdl['Max'])
        return s_mins, s_maxs

    def create_action_space(self, last_actions):
        # available actions are not constant - depend on asset current var
        # spaces = used to define legitimate action space
        actions = []
        for j, asset in enumerate(self.asset_models):
            current = last_actions[j * 2]
            binary = last_actions[j * 2 + 1]
            radius = asset.variables[0]['Radius']
            LB_curr = max(current - radius, asset.variables[0]['Min'])
            UB_curr = min(current + radius, asset.variables[0]['Max'])
            LB_bin = asset.variables[1]['Min']
            UB_bin = asset.variables[1]['Max']
            # only go to minimum load from off
            if binary == 0:
                UB_curr = asset.variables[0]['Min']
            # only turn off if load=min
            if current > asset.variables[0]['Min']:
                LB_bin = asset.variables[1]['Max']
            actions.append([LB_curr, UB_curr])
            actions.append([LB_bin, UB_bin])
        return spaces.MultiDiscrete(actions)

    def asset_mins_maxs(self):
        a_mins, a_maxs = [], []
        for j, asset in enumerate(self.asset_models):
            for var in asset.variables:
                a_mins = np.append(a_mins, var['Min'])
                a_maxs = np.append(a_maxs, var['Max'])
        return a_mins, a_maxs

    def asset_states(self):
        for asset in self.asset_models:
            for var in asset.variables:
                print(var['Name'] + ' is ' + str(var['Current']))
        return self
