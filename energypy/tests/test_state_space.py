import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

from energypy.common.spaces import Space, StateSpace, PrimCfg

#  test data lengths, discretized shape dims, sampling starts


@pytest.mark.parametrize('space', [Space, StateSpace])
@hp.given(prices=st.lists(st.floats(allow_infinity=False), min_size=1, max_size=100))
def test_low_and_high_from_primitives(prices, space):

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


@pytest.mark.parametrize('space', [Space, StateSpace])
@hp.given(prices=st.lists(st.floats(), min_size=1, max_size=100))
def test_num_samples(prices, space):
    space = space('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
    )

    assert space.num_samples == len(prices)


@hp.given(prices=st.lists(st.floats(allow_nan=False), min_size=1, max_size=100))
def test_call_state_space(prices):
    space = StateSpace('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
    )

    for idx in np.random.randint(0, len(prices), size=10):
        state = space(steps=idx, offset=0)
        assert state == prices[idx]


@hp.given(prices=st.lists(st.floats(allow_nan=False), min_size=1, max_size=100))
def test_sample_episode_state_space_full(prices):
    space = StateSpace('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
    )

    start, end = space.sample_episode('full')

    assert start == 0
    assert end == len(prices)


@hp.given(prices=st.lists(st.floats(allow_nan=False), min_size=1, max_size=100))
def test_sample_episode_state_space_random(prices):
    space = StateSpace('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
    )

    starts, ends = [], []
    for _ in range(50):
        start, end = space.sample_episode('random', episode_length=10)
        starts.append(start)
        ends.append(end)

    assert min(starts) >= 0
    assert max(starts) <= len(prices)

    #  catch case when we always start at 0
    if np.mean(starts) != 0:
        #  check we start at different points
        assert np.std(starts) != 0
        #  check we end at different points
        assert np.std(ends) != 0


@hp.given(prices=st.lists(st.floats(allow_nan=False), min_size=1, max_size=100))
def test_sample_episode_state_space_fixed(prices):
    space = StateSpace('state').from_primitives(
        PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
    )

    start, end = space.sample_episode('fixed', 5)

    assert end - start == min(len(prices), 5)
    assert start == 0
