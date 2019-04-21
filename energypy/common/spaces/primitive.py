""" single dimension of a more complex space """

import numpy as np


class Primitive:
    """ inits the primitive spaces """
    def __init__(self, name, low, high, data=None):
        self.name = name
        self.low  = float(low)
        self.high = float(high)
        self.data = np.array(data).reshape(-1)


class ContinuousSpace(Primitive):
    """ single dimension continuous space  - car accelerator """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, steps, offset):
        return float(self.data[steps + offset])

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        cond = (x >= self.low) and (x <= self.high)
        if not cond:
            raise ValueError(
                '{} not in space - min {} max {}'.format(x, self.low, self.high)
            )
        return cond

    def discretize(self, num_discrete):
        return np.linspace(self.low, self.high, num_discrete).tolist()


class DiscreteSpace(Primitive):
    """ single dimension discrete space - gears in car """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = int(self.low)
        self.high = int(self.high)

    def __call__(self, steps, offset):
        #  data is a 1-D array
        return int(self.data[steps + offset])

    def sample(self):
        return np.random.randint(self.high)

    def contains(self, x):
        cond = np.in1d(x, np.arange(0, self.high))[0]
        if not cond:
            raise ValueError
        return cond

    def discretize(self, num_discrete=None):
        return np.arange(0, self.high)
