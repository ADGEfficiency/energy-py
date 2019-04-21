""" test primitive space contains, discretizing and sampling """

import hypothesis as hp
import hypothesis.strategies as st
import pytest

from energypy.common.spaces import ContinuousSpace, DiscreteSpace


@hp.given(
    st.floats(min_value=-100, max_value=0),
    st.floats(min_value=1, max_value=100),
    st.integers(1, 10)
)
def test_continuous(low, high, num_discrete):
    space = ContinuousSpace('state', low, high, None)

    assert space.low == low
    assert space.high == high

    assert space.contains(low)
    assert space.contains((low + high / 2))
    assert space.contains(high)

    with pytest.raises(ValueError):
        assert space.contains(low - 2)
        assert space.contains(high + 2)

    discrete = space.discretize(num_discrete)
    assert len(discrete) == num_discrete


@hp.given(
    st.integers(min_value=6, max_value=100),
)
def test_discrete(high):
    low = 0
    space = DiscreteSpace('state', low, high, None)

    assert space.low == low
    assert space.high == high

    assert space.contains(low)
    assert space.contains(high - 1)

    with pytest.raises(ValueError):
        assert space.contains(-1)
        assert space.contains(low + 0.5)

        assert space.contains(high)
        assert space.contains(high - 0.3)


@hp.given(
    st.floats(min_value=-100, max_value=0),
    st.floats(min_value=1, max_value=100),
)
def test_continuous_sampling(low, high):
    space = ContinuousSpace('state', low, high, None)
    samples = 100
    for _ in range(samples):
        assert space.contains(space.sample())

@hp.given(
    st.integers(min_value=6, max_value=100),
)
def test_discrete_sampling(high):
    low = 0
    space = DiscreteSpace('state', low, high, None)
    samples = 100
    for _ in range(samples):
        assert(space.contains(space.sample()))
