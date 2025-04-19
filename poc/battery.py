import gymnasium as gym
import typing
import numpy as np
import random

import collections


class ExperimentResult:
    pass


class BatteryEnv(gym.Env):
    def __init__(
        self,
        electricity_prices: typing.Sequence[float],
        power_mw=2.0,
        capacity_mwh=4.0,
        efficiency_pct=0.9,
        initial_state_of_charge_mwh=0.0,
        episode_length: int = 48,
    ):
        self.capacity_mwh = capacity_mwh
        self.efficiency_pct: float = efficiency_pct
        self.electricity_prices: typing.Sequence = electricity_prices
        self.episode_length: int = episode_length
        self.index: int = 0
        self.initial_state_of_charge_mwh: float = initial_state_of_charge_mwh
        self.n_lags: int = 20
        assert self.episode_length <= len(self.electricity_prices)

        # lagged prices and current state of charge
        self.observation_space: gym.spaces.Space = gym.spaces.Box(
            low=0, high=1000, shape=(len(electricity_prices) + 1,), dtype=float
        )

        # one action - choose charge / discharge MW for the next interval
        self.action_space = gym.spaces.Discrete(4)

        self.info = collections.defaultdict(list)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        super().reset(seed=seed)
        self.index = random.randint(
            0, len(self.electricity_prices) - self.episode_length
        )
        self.state_of_charge_mwh = self.initial_state_of_charge_mwh
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # TODO - use internal state counter, price data
        # prices with charges stacked on the end
        obs = list(self.electricity_prices[self.index - self.n_lags : self.index]) + [
            self.state_of_charge_mwh
        ]
        obs = np.array(obs, dtype=float).reshape(1, -1)
        assert obs.shape[1] == self.n_lags + 1
        return obs

    def _get_info(self):
        # TODO - some info for experiment analysis (usually)
        return self.info

    def step(self, action: float) -> tuple:
        # TODO - possible this action would be scaled...
        # can i use a wrapper?
        # TODO - not converting from MW to MWh
        battery_power_mw = action

        interval_initial_state_of_charge_mwh = self.state_of_charge_mwh
        interval_final_state_of_charge_mwh = np.clip(
            # TODO - not converting from MW to MWh
            interval_initial_state_of_charge_mwh + battery_power_mw,
            0,
            self.capacity_mwh,
        )

        # TODO losses - should come off the delta charge
        interval_net_charge_mwh = (
            interval_final_state_of_charge_mwh - interval_initial_state_of_charge_mwh
        )

        reward = self.electricity_prices[self.index] * battery_power_mw

        terminated = self.index == self.episode_length
        self.index += 1
        self.state_of_charge_mwh = interval_final_state_of_charge_mwh
        self.info["state_of_charge_mwh"].append(self.state_of_charge_mwh)
        return self._get_obs(), reward, terminated, False, self._get_info()


env_id = "energypy/battery"
gym.register(
    id=env_id,
    entry_point=BatteryEnv,
)

# TODO - make into a test
print(gym.pprint_registry())
env = gym.make(env_id, electricity_prices=np.random.uniform(-1000, 1000, 2**8))
env = gym.wrappers.NormalizeReward(env)
print(env.reset())
for _ in range(20):
    o, r, d, t, i = env.step(10)
    print(r)


class BatteryVectorEnv(gym.vector.VectorEnv):
    pass


# add that to docs folders
