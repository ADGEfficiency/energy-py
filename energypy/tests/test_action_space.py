""" test action space """

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

from energypy.common.spaces import ActionSpace, PrimitiveConfig

action = ActionSpace().from_primitives(
    PrimitiveConfig('action', 0, 100, 'continuous', None),
    PrimitiveConfig('action1', 0, 20, 'discrete', None)
)

discrete = action.discretize(20)

assert len(discrete) == 20 ** (len(action.keys()))


