from collections import defaultdict

import numpy as np
import pytest

from energypy.registry import make


#  cfg, actions, expected_losses
test_cases = (
    #  full charge for two steps, full discharge for two steps
    (
        {
            'initial_charge': 0.0,
            'power': 2.0,
            'capacity': 100,
            'episode_length': 4,
            'efficiency': 0.9
        },
        [1.0, 1.0, -1.0, -1.0],
        [0, 0, 0.2, 0.2]
    ),
    (
        {'initial_charge': 1.0, 'power': 3.0, 'capacity': 100, 'episode_length': 4, 'efficiency': 0.8},
        [-0.5, -0.5, -0.5, -0.5],
        [3 * 0.5 * (1-0.8)] * 4
    ),
)

@pytest.mark.parametrize('cfg, actions, expected_losses', test_cases)
def test_one_battery_charging(cfg, actions, expected_losses):
    env = make('battery', **cfg, n_batteries=1)
    env.reset()

    results = defaultdict(list)
    for action in actions:
        action = np.array(action).reshape(1, 1)
        next_obs, reward, done, info = env.step(action)
        results['losses'].append(info['losses_power'])
        results['gross_power'].append(info['gross_power'])

    assert done
    losses = np.squeeze(np.array(results['losses']))
    import pandas as pd
    print(pd.DataFrame(results))
    np.testing.assert_array_almost_equal(losses, expected_losses)
