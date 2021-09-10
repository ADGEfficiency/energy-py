from collections import namedtuple
import numpy as np

from energypy import registry
from energypy.envs.base import AbstractEnv


def battery_energy_balance(
    initial_charge, final_charge, import_energy, export_energy, losses
):
    delta_charge = final_charge - initial_charge
    balance = import_energy - (export_energy + delta_charge + losses)
    np.testing.assert_almost_equal(balance, 0)


def calculate_losses(delta_charge, efficiency):
    delta_charge = np.array(delta_charge)
    efficiency = np.array(efficiency)

    #  account for losses / the round trip efficiency
    #  we lose electricity when we discharge
    losses = delta_charge * (1 - efficiency)
    losses = np.array(losses)
    losses[delta_charge > 0] = 0

    # if (np.isnan(losses)).any():
    #     losses = np.zeros_like(losses)
    return np.abs(losses)


def set_battery_config(value, n_batteries):
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return np.array(value).reshape(n_batteries, 1)
    elif isinstance(value, np.ndarray):
        return np.array(value).reshape(n_batteries, 1)
    else:
        return np.full((n_batteries, 1), value).reshape(n_batteries, 1)


class BatteryObservationSpace:
    def __init__(self, dataset, additional_features):
        shape = list(dataset.episode["features"].shape[2:])
        shape[-1] += additional_features
        self.shape = tuple(shape)

    def get_mask_shape(self):
        return (self.shape[0], self.shape[0])


class BatteryActionSpace:
    def __init__(self, n_batteries=2):
        self.n_batteries = n_batteries
        self.shape = (1,)

        self.low = -1
        self.high = 1

    def sample(self):
        return np.random.uniform(-1, 1, self.n_batteries).reshape(self.n_batteries, 1)

    def contains(self, action):
        assert (action <= 1.0).all()
        assert (action >= -1.0).all()
        return True


class Battery(AbstractEnv):
    """
    data = (n_battery, timesteps, features)
    """

    def __init__(
        self,
        n_batteries=2,
        power=2.0,
        capacity=4.0,
        efficiency=0.9,
        initial_charge=0.0,
        episode_length=288,
        dataset={"name": "random-dataset"},
        logger=None,
    ):
        self.n_batteries = n_batteries

        #  2 = half hourly, 6 = 5 min
        self.timestep = 2

        #  kW
        self.power = set_battery_config(power, n_batteries)
        #  kWh
        self.capacity = set_battery_config(capacity, n_batteries)
        #  %
        self.efficiency = set_battery_config(efficiency, n_batteries)
        #  kWh
        initial_charge = np.clip(initial_charge, 0, 1.0)
        self.initial_charge = set_battery_config(initial_charge * capacity, n_batteries)

        self.episode_length = int(episode_length)

        if isinstance(dataset, dict):
            self.dataset = registry.make(
                **dataset, logger=logger, n_batteries=n_batteries
            )
        else:
            assert dataset.n_batteries == self.n_batteries
            self.dataset = dataset

        self.reset("train")

        self.observation_space = BatteryObservationSpace(
            self.dataset, additional_features=1
        )
        self.action_space = BatteryActionSpace(n_batteries)

        mask_shape = self.observation_space.get_mask_shape()

        self.elements = (
            ("observation", self.observation_space.shape, "float32"),
            ("action", self.action_space.shape, "float32"),
            ("reward", (1,), "float32"),
            ("next_observation", self.observation_space.shape, "float32"),
            ("done", (1,), "bool"),
            #  attention specific - TODO toggle these out for non attention
            ("observation_mask", mask_shape, "float32"),
            ("next_observation_mask", mask_shape, "float32"),
        )

        self.Transition = namedtuple("Transition", [el[0] for el in self.elements])

    def reset(self, mode="train"):
        self.cursor = 0
        self.charge = self.get_initial_charge()

        self.dataset.reset(mode)
        self.test_done = self.dataset.test_done
        return self.get_observation()

    def get_initial_charge(self):
        #  instance check to avoid a warning that occurs when initial_charge is an array
        if isinstance(self.initial_charge, str) and self.initial_charge == "random":
            initial = np.random.uniform(0, self.capacity[0], self.n_batteries)
        else:
            initial = self.initial_charge
        return initial.reshape(self.n_batteries, 1)

    def get_observation(self):
        """one timestep"""
        data = self.dataset.sample_observation(self.cursor)
        features = data["features"]

        #  adding the charge onto the observation
        #  different depending on attention or not

        if len(features.shape) == 2:
            #  (n_batteries, n_features)
            features = data["features"].reshape(self.n_batteries, -1)
            features = np.concatenate([features, self.charge], axis=1)
        else:
            #  (n_batteries, sequence_length, n_features)
            assert len(features.shape) == 3
            sh = features.shape
            assert sh[0] == self.n_batteries

            #  (batch, n_batteries, sequence_length, n_features)
            sequence_length = sh[1]
            features = data["features"].reshape(
                self.n_batteries, sequence_length, sh[2]
            )

            #  TODO
            #  we only have charge for one timestep (but many batteries)
            #  but our features are across many timesteps
            #  solution here is to tile charge - in reality this should go into a different
            #  part of the network (multihead network)

            chg = self.charge.reshape(self.n_batteries, 1, 1)
            chg = np.repeat(chg, sequence_length, axis=1)
            #  concat along the features axis
            features = np.concatenate([features, chg], axis=2)

            #  (batch, n_batteries, sequence_length, sequence_length)
            mask = data["mask"].reshape(
                self.n_batteries, sequence_length, sequence_length
            )

        return {"features": features, "mask": mask}

    def setup_test(self):
        self.test_done = self.dataset.setup_test()

    def step(self, action):
        action_power = action.reshape(self.n_batteries, 1)

        #  expect a scaled action here
        #  -1 = discharge max, 1 = charge max
        action_power = np.clip(action_power, -1, 1)
        action_power = action_power * self.power

        #  convert from power to energy, kW -> kWh
        action_energy = action_power / self.timestep

        #  charge at the start of the interval, kWh
        initial_charge = self.charge

        #  charge at end of the interval
        #  clipped at battery capacity, kWh
        final_charge = np.clip(initial_charge + action_energy, 0, self.capacity)

        #  accumulation in battery, kWh
        #  delta_charge can also be thought of as gross_power
        delta_charge = final_charge - initial_charge

        #  losses are applied when we discharge, kWh
        losses = calculate_losses(delta_charge, self.efficiency)

        #  net of losses, kWh
        #  add losses here because in delta_charge, export is negative
        #  to reduce export, we add a positive losses
        net_energy = delta_charge + losses

        import_energy = np.zeros_like(net_energy)
        import_energy[net_energy > 0] = net_energy[net_energy > 0]

        export_energy = np.zeros_like(net_energy)
        export_energy[net_energy < 0] = np.abs(net_energy[net_energy < 0])

        #  set charge for next timestep
        self.charge = initial_charge + delta_charge

        #  check battery is working correctly
        battery_energy_balance(
            initial_charge, self.charge, import_energy, export_energy, losses
        )

        price = self.dataset.sample_observation(self.cursor)["prices"].reshape(
            self.n_batteries, -1
        )
        price = np.array(price).reshape(self.n_batteries, 1)
        reward = export_energy * price - import_energy * price

        self.cursor += 1
        done = np.array(self.cursor == (self.episode_length))

        next_obs = self.get_observation()

        info = {
            "cursor": self.cursor,
            "episode_length": self.episode_length,
            "done": done,
            "gross_power": delta_charge * self.timestep,
            "net_power": net_energy * self.timestep,
            "losses_power": losses * self.timestep,
            "initial_charge": initial_charge,
            "final_charge": self.charge,
        }

        return next_obs, reward, done, info
