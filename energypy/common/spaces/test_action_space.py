""" test action space """

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

from energypy.common.spaces import ActionSpace, PrimCfg

action = ActionSpace().from_primitives(
    PrimCfg('action', 0, 100, 'continuous', None),
    PrimCfg('action1', 0, 20, 'discrete', None)
)

discrete = action.discretize(20)

assert len(discrete) == 20 ** (len(action.keys()))


