import energypy

default_config = {
    'env_id': 'battery',
     'dataset': 'example',
     'initial_charge': 0,
     'round_trip_eff': 0.9,
    'episode_sample': 'full'
}

env = energypy.make_env(**default_config)
obs = env.reset()


def test_charge():

    rew, next_obs, d, i = env.step(1.0)

    charge = env.get_state_variable('C_charge_level [MWh]')

    expected_charge = 0.9 * 1.0 / 12

    assert charge == expected_charge


def test_discharge():
    config = default_config
    config['initial_charge'] = 1.0
    config['capacity'] = 4.0

    env = energypy.make_env(**config)
    obs = env.reset()

    rew, next_obs, d, i = env.step(-1.0)

    charge = env.get_state_variable('C_charge_level [MWh]')

    expected_charge = 4.0 - 1.0 / 12

    assert charge == expected_charge


def test_no_op():
    config = default_config
    config['initial_charge'] = 0.5
    config['capacity'] = 4.0

    env = energypy.make_env(**config)
    obs = env.reset()

    rew, next_obs, d, i = env.step(0)

    charge = env.get_state_variable('C_charge_level [MWh]')

    expected_charge = 2.0

    assert charge == expected_charge
