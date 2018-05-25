import numpy as np


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find_sub_array_in_2D_array(sub_array, array):
    """
    Find the first occurence of a sub_array within a larger array

    args
        sub_array (np.array) ndim=1
        array (np.array) ndim=2, shape=(num_samples, sub_array.shape[0])

    i.e. 
        sub_array = np.array([0.0, 2.0]).reshape(2)
        array = np.array([0.0, 0.0,
                          0.0, 1.0,
                          0.0, 2.0).reshape(3, 2)
        --> 2

    Used for finding the index of an action within a list of all possible actions
    """
    assert sub_array.ndim == 1
    assert array.ndim == 2

    bools = rolling_window(sub_array, array.shape[1]) == array

    bools = np.all(
        bools.reshape(array.shape[0], -1),
        axis=1
    )

    #  argmax finds the first true values
    return np.argmax(bools)


