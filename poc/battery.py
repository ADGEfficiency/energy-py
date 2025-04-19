import gymnasium as gym
import typing
import numpy as np
import random

import collections


class ExperimentResult:
    pass


def battery_energy_balance(
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


class BatteryEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        electricity_prices: typing.Sequence[float],
        power_mw=2.0,
        capacity_mwh=4.0,
        efficiency_pct=0.9,
        initial_state_of_charge_mwh: float = 0.0,
        episode_length: int = 48,
    ):
        self.capacity_mwh = capacity_mwh
        self.efficiency_pct: float = efficiency_pct
        self.electricity_prices: typing.Sequence[float] = electricity_prices
        self.episode_length: int = episode_length
        self.index: int = 0
        self.initial_state_of_charge_mwh: float = initial_state_of_charge_mwh
        self.n_lags: int = 0
        self.n_horizons: int = 48
        assert self.episode_length + self.n_lags <= len(self.electricity_prices)

        # lagged prices and current state of charge
        self.observation_space: gym.spaces.Space[np.ndarray] = gym.spaces.Box(
            low=0, high=1000, shape=(self.n_lags + self.n_horizons + 1,)
        )

        # one action - choose charge / discharge MW for the next interval
        self.action_space = gym.spaces.Box(low=-power_mw, high=power_mw)

        self.info = collections.defaultdict(list)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        super().reset(seed=seed)
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

    def _get_obs(self):
        # TODO - use internal state counter, price data
        # prices with charges stacked on the end
        obs = list(
            self.electricity_prices[
                self.index - self.n_lags : self.index + self.n_horizons
            ]
        ) + [self.state_of_charge_mwh]
        obs = np.array(obs, dtype=float)
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
        gross_charge_mwh = np.clip(
            # TODO - not converting from MW to MWh
            interval_initial_state_of_charge_mwh + battery_power_mw,
            0,
            self.capacity_mwh,
        )

        # TODO losses - should come off the delta charge
        losses = 0
        interval_net_charge_mwh = gross_charge_mwh - losses
        interval_final_state_of_charge_mwh = (
            interval_initial_state_of_charge_mwh + interval_net_charge_mwh
        )

        # TODO units
        import_energy = interval_net_charge_mwh if interval_net_charge_mwh > 0 else 0
        export_energy = interval_net_charge_mwh if interval_net_charge_mwh < 0 else 0

        battery_energy_balance(
            initial_charge=interval_initial_state_of_charge_mwh,
            final_charge=interval_final_state_of_charge_mwh,
            import_energy=import_energy,
            export_energy=export_energy,
            losses=losses,
        )

        # TODO import & export prices
        reward = self.electricity_prices[self.index] * battery_power_mw
        terminated = self.episode_step + 1 == self.episode_length

        # print(terminated, self.index, self.episode_length)
        self.index += 1
        self.episode_step += 1
        self.state_of_charge_mwh = float(interval_final_state_of_charge_mwh)
        self.info["state_of_charge_mwh"].append(self.state_of_charge_mwh)
        return self._get_obs(), reward, terminated, False, self._get_info()


env_id = "energypy/battery"
gym.register(
    id=env_id,
    entry_point=BatteryEnv,
)

# TODO - make into a test
print(gym.pprint_registry())
env = gym.make(env_id, electricity_prices=np.random.uniform(-1000, 1000, 10000))
env = gym.wrappers.NormalizeReward(env)
# print(env.reset())
# for _ in range(20):
#     o, r, d, t, i = env.step(10)
#     print(r)

from energypy.runner import main

from stable_baselines3 import PPO

main(
    env=env,
    eval_env=env,
    model=PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    ),
    name="cartpole",
)
