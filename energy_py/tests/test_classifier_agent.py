from collections import namedtuple

import numpy as np

from energy_py.agents import ClassifierStragety, ClassifierCondition


string = 'D_h_{}_Predicted_Price_Bin_{}'
bins = ['High', 'Very High', 'High', 'Very High']
horizions = [0, 0, 6, 6]
obs_info = [string.format(h, b) for h, b in zip(horizions, bins)]


def test_stragety():

    strat = ClassifierStragety(conditions=[ClassifierCondition(0, 'Very High', '=='),
                                 ClassifierCondition(6, 'Very High', '!=')],
                     action=np.array(2),
                     observation_info=obs_info)

    obs = np.array([0, 1, 1, 0]).reshape(1, 4)

    assert strat.compare_condition(obs, strat.conditions[0])
    assert strat.compare_condition(obs, strat.conditions[1])

    action = strat.check_observation(obs)

    assert action == np.array(2)

def test_no_op():

    strat = ClassifierStragety(conditions=[ClassifierCondition(0, 'Very High', '=='),
                                 ClassifierCondition(6, 'Very High', '!=')],
                     action=np.array(1),
                     observation_info=obs_info)

    obs = np.array([1, 0, 1, 0]).reshape(1, 4)

    action = strat.check_observation(obs)

    assert action == np.array(0)
