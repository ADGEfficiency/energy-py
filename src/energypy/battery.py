import collections
import random
import typing

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


# Define a Protocol for objects that have a shape attribute
class HasShape(typing.Protocol):
    shape: typing.Any


# Use Union with explicit types to ensure proper type checking
NumericSequence = typing.Union[NDArray[np.float64], typing.Sequence[float]]


class Freq:
    """Handles conversion of power (MW) to energy (MWh) at different interval frequencies."""

    def __init__(self, mins: int) -> None:
        """Initialize a Freq class."""
        self.mins = mins

    def mw_to_mwh(self, mw: float) -> float:
        """Convert power MW to energy MWh."""
        return mw * self.mins / 60

    def mwh_to_mw(self, mw: float) -> float:
        """Convert energy MWh to power MW."""
        return mw * 60 / self.mins

    def __repr__(self) -> str:
        """Control printing."""
        return f"Freq(mins={self.mins})"


class Battery(gym.Env[NDArray[np.float64], NDArray[np.float64]]):
    def __init__(
        self,
        electricity_prices: NumericSequence = np.random.uniform(-100.0, 100, 48 * 10),
        features: NDArray[np.float64] = np.random.uniform(-100.0, 100, (48 * 10, 4)),
        power_mw: float = 2.0,
        capacity_mwh: float = 4.0,
        efficiency_pct: float = 0.9,
        initial_state_of_charge_mwh: float = 0.0,
        episode_length: int = 48,
        freq_mins: int = 60,  # Default to hourly
    ):
        self.power_mw = power_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency_pct = efficiency_pct
        self.electricity_prices = electricity_prices
        self.features = features
        self.freq = Freq(mins=freq_mins)

        assert len(self.electricity_prices) == features.shape[0], (
            "Features and prices must have same length"
        )
        self.n_features = features.shape[1]

        self.episode_length: int = episode_length
        self.index: int = 0
        self.initial_state_of_charge_mwh: float = initial_state_of_charge_mwh
        self.n_lags: int = 0
        self.n_horizons: int = 48
        self.episode_step: int = 0
        self.state_of_charge_mwh: float = initial_state_of_charge_mwh
        assert self.episode_length + self.n_lags <= len(self.electricity_prices)

        # Observation space includes features and current state of charge
        self.observation_space: gym.spaces.Space[NDArray[np.float64]] = gym.spaces.Box(
            low=-1000, high=1000, shape=(self.n_features + 1,), dtype=np.float64
        )

        # one action - choose charge / discharge MW for the next interval
        self.action_space = gym.spaces.Box(
            low=-power_mw, high=power_mw, shape=(1,), dtype=np.float32
        )

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
        # Get features for the current time step
        feature_obs = self.features[self.index].tolist()
        # Add state of charge to observation
        obs = feature_obs + [self.state_of_charge_mwh]
        return np.array(obs, dtype=np.float64)

    def _get_info(self) -> dict[str, list[float]]:
        # Include current price and feature values in info
        return self.info

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, list[float]]]:
        # clip action to battery power limits
        battery_power_mw = float(np.clip(action, -self.power_mw, self.power_mw)[0])

        # convert power (MW) to energy (MWh) based on frequency interval
        energy_change_mwh = self.freq.mw_to_mwh(battery_power_mw)

        initial_charge_mwh = self.state_of_charge_mwh
        final_charge_mwh = float(
            np.clip(initial_charge_mwh + energy_change_mwh, 0, self.capacity_mwh)
        )

        gross_charge_mwh = final_charge_mwh - initial_charge_mwh

        # Define import and export energy based on charge direction
        import_energy_mwh = 0.0
        export_energy_mwh = 0.0
        losses = 0.0

        if gross_charge_mwh > 0:  # Charging
            # No losses during charging
            import_energy_mwh = gross_charge_mwh
        elif gross_charge_mwh < 0:  # Discharging
            # When discharging, we lose energy during conversion
            # The SOC decreases by gross_charge_mwh (which is negative)
            # But due to efficiency losses, the exported energy is less than the SOC decrease
            energy_removed_from_storage = abs(gross_charge_mwh)
            export_energy_mwh = energy_removed_from_storage * self.efficiency_pct
            losses = energy_removed_from_storage - export_energy_mwh

        self.energy_balance(
            initial_charge=initial_charge_mwh,
            final_charge=float(final_charge_mwh),
            import_energy=float(import_energy_mwh),
            export_energy=float(export_energy_mwh),
            losses=losses,
        )

        # Calculate reward using price
        reward = float(self.electricity_prices[self.index] * export_energy_mwh) - float(
            self.electricity_prices[self.index] * import_energy_mwh
        )
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
        # For energy balance:
        # When charging: import = delta_charge (no losses)
        # When discharging: delta_charge (negative) + export + losses = 0
        delta_charge = final_charge - initial_charge

        if delta_charge > 0:  # Charging
            # import_energy = delta_charge
            balance = import_energy - delta_charge
        else:  # Discharging or no change
            # export_energy + losses = -delta_charge
            balance = export_energy + losses + delta_charge

        # print(
        #     f"battery_energy_balance: {initial_charge=}, {final_charge=}, {import_energy=}, {export_energy=}, {losses=}, {balance=}"
        # )
        np.testing.assert_allclose(balance, 0, atol=0.00001)

        assert final_charge <= self.capacity_mwh, (
            f"battery-capacity-constraint: {final_charge=}, {self.capacity_mwh=}"
        )
