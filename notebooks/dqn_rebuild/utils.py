import numpy as np
import tensorflow as tf


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
    #  array making and shaping so that user could feed in a list and it
    #  would work
    sub_array = np.array(sub_array).reshape(array.shape[1])

    bools = rolling_window(sub_array, array.shape[1]) == array

    bools = np.all(
        bools.reshape(array.shape[0], -1),
        axis=1
    )

    #  argmax finds the first true values
    return np.argmax(bools)


def get_tf_params(scope):
    params = [p for p in tf.trainable_variables()
              if p.name.startswith(scope)]

    #  sort parameters list by the variable name
    params = sorted(params, key=lambda var: var.name)

    return params


def make_copy_ops(parent, child, scope='copy_ops'):
    copy_ops = []

    with tf.variable_scope(scope):
        tau = tf.Variable(1.0, name='tau')

        for p, c in zip(parent, child):
            assert p.name.split('/')[1:] == c.name.split('/')[1:]

            new_value = tf.add(tf.multiply(p, tau),
                               tf.multiply(c, 1 - tau))
            op = c.assign(new_value)
            copy_ops.append(op)

    return copy_ops, tau
