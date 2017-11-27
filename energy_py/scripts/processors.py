"""
Purpose of this script is to hold Processor objects - a class of objects
used for functionality such as standardization or normalization.

We can allow the class to collect statistics on the data it has seen so far.

We can also choose to process a batch using statistics only from that batch.
"""

import numpy as np

#  use epsilon to catch div/0 errors
epsilon = 1e-5

class Standardizer(object):
    """
    We rely on the input shape being (n_samples, state_dim)
    """
    def __init__(self, use_history=True, space=[]):
        #  use a list to hold all data this processor has seen
        self.history = []

        #  if we include a space object then we use it to fill up the history
        #  and we always use history
        if space:
            self.history = [space.sample() for i in range(1000)]
            use_history = True

        self.use_history = use_history

    def transform(self, batch):
        """
        """
        #  check that our data is
        assert batch.ndim == 2
        #  add the data we are processing onto our history list
        self.history.append(batch)

        if self.use_history:
            #  create an array from the list then reshape to (num_samples, dim)
            #  taking advantage of energy_py states/actions being this shape (always len 2)
            history = np.concatenate(self.history).reshape(-1, batch.shape[1])
            means, stdevs = history.mean(axis=0), history.std(axis=0)
        else:
            means, stdevs = batch.mean(axis=0), batch.std(axis=0)

        return (batch - means) / (stdevs + epsilon)


class Normalizer(object):
    """
    We rely on the input shape being (n_samples, state_dim)
    """
    def __init__(self, use_history=True, space=[]):
        #  use a list to hold all data this processor has seen
        self.history = []

        #  if we include a space object we use this and don't ever use history
        if space:
            self.mins = np.array([spc.low for spc in space.spaces])
            self.maxs = np.array([spc.high for spc in space.spaces])
            use_history = False

        self.use_history = use_history

    def transform(self, batch):
        """
        """
        assert batch.ndim == 2
        #  add the data we are processing onto our history list
        self.history.append(batch)

        if self.use_history:
            #  create an array from the list then reshape to (num_samples, dim)
            #  taking advantage of energy_py states/actions being this shape (always len 2)
            history = np.concatenate(self.history).reshape(-1, batch.shape[1])
            self.maxs, self.mins = history.max(axis=0), history.min(axis=0)
        else:
            self.maxs, self.mins = batch.max(axis=0), batch.min(axis=0)

        return (batch - self.mins) / (self.maxs - self.mins + epsilon)
