""" test global space """

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

from energypy.common.spaces import Space, StateSpace, PrimCfg


@pytest.mark.parametrize('space', [Space, StateSpace])
@hp.given(prices=st.lists(st.floats(), min_size=1, max_size=100))
def test_low_and_high(prices, space):

    state = space('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
        PrimCfg('charge [MWh]', 0, 50, 'discrete', None)
    )

    assert list(state.keys()) == ['price [$/MWh]', 'charge [MWh]']

    np.testing.assert_array_equal(state.high, [max(prices), 50])
    np.testing.assert_array_equal(state.low, [min(prices), 0])


@pytest.mark.parametrize('space', [Space, StateSpace])
def test_sample_and_contains(space):

    prices = np.random.rand(1000)

    space = space('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
        PrimCfg('charge [MWh]', 0, 50, 'continuous', None)
    )

    for _ in range(100):
        assert space.contains(space.sample())


