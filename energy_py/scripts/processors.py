"""
Purpose of this script is to hold Processor objects - a class of objects
used for functionality such as standardization or normalization.

We can allow the class to collect statistics on the data it has seen so far.

We can also choose to process a batch using statistics only from that batch.
"""

import numpy as np


class Standardizer(object):
    """
    We rely on the input shape being (n_samples, state_dim)
    """
    def __init__(self):
        #  use a list to hold all data this processor has seen
        self.history = []

    def transform(self, batch, use_history=True):
        """
        The SimpleStand:ardizer transforms the data using the mean and standard
        deviation across the batch it is processing
        """
        #  check that our data is
        assert len(batch.shape) == 2
        #  add the data we are processing onto our history list
        self.history.append(batch)
        if use_history:
            #  create an array from the list then reshape to (num_samples, dim)
            #  taking advantage of energy_py states/actions being this shape (always len 2)
            history = np.array(self.history).reshape(-1, batch.shape[1])
            means, stdevs = history.mean(axis=0), history.std(axis=0)
        else:
            means, stdevs = batch.mean(axis=0), batch.std(axis=0)

        return (batch - means) / stdevs


class Normalizer(object):
    """
    We rely on the input shape being (n_samples, state_dim)
    """
    def __init__(self):
        #  use a list to hold all data this processor has seen
        self.history = []

    def transform(self, batch):
        """
        The SimpleStandardizer transforms the data using the mean and standard
        deviation across the batch it is processing
        """
        assert len(batch.shape) == 2
        #  add the data we are processing onto our history list
        self.history.append(batch)
        #  create an array from the list then reshape to (num_samples, dim)
        #  taking advantage of energy_py states/actions being this shape (always len 2)
        history = np.array(self.history).reshape(-1, batch.shape[1])

        return (batch - history.mean(axis=0)) / history.std(axis=0)
