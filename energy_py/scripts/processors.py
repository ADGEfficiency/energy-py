"""
Purpose of this script is to hold Processor objects - a class of objects
used for functionality such as standardization or normalization.

We can allow the class to collect statistics on the data it has seen so far.

We can also choose to process a batch using statistics only from that batch.

"""

import numpy as np

#  use epsilon to catch div/0 errors
epsilon = 1e-5

class Processor(object):
    """
    A base class for Processor objects

    args
        length (int): the length of the array
                      this corresponds to the second dimension of the shape
                      shape = (num_samples, length)
        use_history (bool): whether to use the history to estimtate parameters
        global_space (GlobalSpace): used to initialize history
    """
    def __init__(self, length, use_history=True, space=[]):
        self.length = length
        self.history = []

    def transform(self, batch):
        """
        Transforms a single array (shape=(length,) 
        or a batch (shape=(num_samples, length))

        args
            batch (np.array)

        returns
            transformed_batch (np.array): shape = (num_samples, length)
        """
        #  catch the case where we process a single obs of shape (obs_dim,) 
        if batch.ndim != 2:
            batch = batch.reshape(-1, self.length)
        assert batch.ndim == 2

        #  add the data we are processing onto our history list
        self.history.append(batch)
        return self._transform(batch)

class Standardizer(Processor):
    """
    Processor object for performing standardization

    Standardization = scaling for zero mean, unit variance
    """
    def __init__(self, length, use_history=True, space=[]):
        super().__init__(length)

        #  if we include a space object then we use it to fill up the history
        #  and we always use history
        if space:
            self.history = [space.sample() for i in range(1000)]
            use_history = True

        self.use_history = use_history

    def _transform(self, batch):

        if self.use_history:
            #  create an array from history 
            #  then reshape to (num_samples, space length)
            history = np.concatenate(self.history).reshape(-1, batch.shape[1])
            means, stdevs = history.mean(axis=0), history.std(axis=0)

        else:
            means, stdevs = batch.mean(axis=0), batch.std(axis=0)

        return (batch - means) / (stdevs + epsilon)


class Normalizer(Processor):
    """
    Processor object for performing normalization to range [0,1]

    Normalization = (val - min) / (max - min) 
    """
    def __init__(self, length, use_history=True, space=[]):
        super().__init__(length)

        #  if we include a space object we use this and don't ever use history
        if space:
            self.mins = np.array([spc.low for spc in space.spaces])
            self.maxs = np.array([spc.high for spc in space.spaces])
            use_history = False

        self.use_history = use_history

    def _transform(self, batch):
        if self.use_history:
            #  create an array from the list then reshape to (num_samples, dim)
            #  taking advantage of energy_py states/actions being this shape (always len 2)
            history = np.concatenate(self.history).reshape(-1, batch.shape[1])
            self.maxs, self.mins = history.max(axis=0), history.min(axis=0)
        else:
            self.maxs, self.mins = batch.max(axis=0), batch.min(axis=0)

        return (batch - self.mins) / (self.maxs - self.mins + epsilon)
