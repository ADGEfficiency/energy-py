import collections
import random
import typing

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

NumericSequence = NDArray[np.float64] | typing.Sequence[float]


class Battery(gym.Env[NDArray[np.float64], NDArray[np.float64]]):
    def __init__(
        self,
        electricity_prices: NumericSequence = np.random.uniform(-100.0, 100, 48 * 10),
        features: NumericSequence = np.random.uniform(-100.0, 100, (48 * 10, 4)),
        power_mw=2.0,
        capacity_mwh=4.0,
        efficiency_pct=0.9,
        initial_state_of_charge_mwh: float = 0.0,
        episode_length: int = 48,
    ):
        self.power_mw = power_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency_pct: float = efficiency_pct
        self.electricity_prices: NumericSequence = electricity_prices
        # TODO - USE FEATURES!!!

        self.episode_length: int = episode_length
        self.index: int = 0
        self.initial_state_of_charge_mwh: float = initial_state_of_charge_mwh
        self.n_lags: int = 0
        self.n_horizons: int = 48
        self.episode_step: int = 0
        self.state_of_charge_mwh: float = initial_state_of_charge_mwh
        assert self.episode_length + self.n_lags <= len(self.electricity_prices)

        # lagged prices and current state of charge
        self.observation_space: gym.spaces.Space[NDArray[np.float64]] = gym.spaces.Box(
            low=-1000, high=1000, shape=(self.n_lags + self.n_horizons + 1,)
        )

        # one action - choose charge / discharge MW for the next interval
        self.action_space = gym.spaces.Box(low=-power_mw, high=power_mw)

        self.info: dict[str, list[float]] = collections.defaultdict(list)

    def reset(
        self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
    ) -> tuple[NDArray[np.float64], dict[str, list[float]]]:
        super().reset(seed=seed, options=options)
        self.index = random.randint(
            self.n_lags + self.episode_length + self.n_horizons,
            len(self.electricity_prices)
            - self.episode_length
            - self.n_lags
            - self.n_horizons,
        )
        self.episode_step = 0
        self.state_of_charge_mwh = self.initial_state_of_charge_mwh
        # print(f"reset: {self.index=}")
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> NDArray[np.float64]:
        # TODO - use internal state counter, price data
        # prices with charges stacked on the end
        obs = list(
            self.electricity_prices[
                self.index - self.n_lags : self.index + self.n_horizons
            ]
        ) + [self.state_of_charge_mwh]
        obs = np.array(obs, dtype=np.float64)
        return obs

    def _get_info(self) -> dict[str, list[float]]:
        # TODO - some info for experiment analysis (usually)
        return self.info

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, list[float]]]:
        # TODO - possible this action would be scaled...
        # can i use a wrapper?

        # TODO - not converting from MW to MWh
        battery_power_mw = np.clip(action, -self.power_mw, self.power_mw)

        initial_charge_mwh = self.state_of_charge_mwh
        final_charge_mwh = np.clip(
            initial_charge_mwh + battery_power_mw, 0, self.capacity_mwh
        )

        gross_charge_mwh = final_charge_mwh - initial_charge_mwh
        losses = 0
        net_charge_mwh = gross_charge_mwh - losses

        import_energy_mwh = net_charge_mwh if net_charge_mwh > 0 else 0
        export_energy_mwh = np.abs(net_charge_mwh) if net_charge_mwh < 0 else 0

        self.energy_balance(
            initial_charge=initial_charge_mwh,
            final_charge=float(final_charge_mwh),
            import_energy=float(import_energy_mwh),
            export_energy=float(export_energy_mwh),
            losses=losses,
        )

        # TODO import & export prices
        reward = float(self.electricity_prices[self.index] * battery_power_mw)
        terminated = self.episode_step + 1 == self.episode_length
        truncated = False

        # print(terminated, self.index, self.episode_length)
        self.index += 1
        self.episode_step += 1
        self.state_of_charge_mwh = float(final_charge_mwh)
        self.info["state_of_charge_mwh"].append(self.state_of_charge_mwh)
        self.info["battery_power_mw"].append(float(battery_power_mw))
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def energy_balance(
        self,
        initial_charge: float,
        final_charge: float,
        import_energy: float,
        export_energy: float,
        losses: float,
    ) -> None:
        delta_charge = final_charge - initial_charge
        balance = import_energy - (export_energy + delta_charge + losses)
        # print(
        #     f"battery_energy_balance: {initial_charge=}, {final_charge=}, {import_energy=}, {export_energy=}, {losses=}, {balance=}"
        # )
        np.testing.assert_allclose(balance, 0, atol=0.00001)

        assert final_charge <= self.capacity_mwh, (
            f"battery-capacity-constraint: {final_charge=}, {self.capacity_mwh=}"
        )
