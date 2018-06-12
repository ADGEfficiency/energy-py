"""Helper functions for tensorflow"""

import tensorflow as tf


def get_tf_params(scope):
    """
    Makes a list of all tf.Variables under this scope

    args
        scope (str)

    returns
        params (list)
    """
    #  find scope parameters
    params = [p for p in tf.trainable_variables()
              if p.name.startswith(scope)]

    #  sort parameters list by the variable name
    return sorted(params, key=lambda var: var.name)


def make_copy_ops(parent, child, scope='copy_ops'):
    """
    Creates the operations to copy variables 

    args
        parent (list of tf.Variables)
        child (list of tf.Variables)

    returns
        copy_ops (list of tf.Operations)
        tau (tf.placeholder)
    """
    with tf.variable_scope(scope):
        tau = tf.Variable(1.0, name='tau')

        copy_ops = []
        for p, c in zip(parent, child):
            assert p.name.split('/')[1:] == c.name.split('/')[1:]

            new_value = tf.add(tf.multiply(p, tau),
                               tf.multiply(c, 1 - tau))

            op = c.assign(new_value)

            copy_ops.append(op)

    return copy_ops, tau
