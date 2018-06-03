import numpy as np
import tensorflow as tf

from utils import make_copy_ops


def make_vars(num):
    """
    Helper function to create tf varaibles
    """
    variables = []
    for n in range(num):
        variables.append(tf.get_variable(
            'var_{}'.format(n),
            shape=(6,),
            initializer=tf.initializers.random_normal
        )
                         )
    return variables


def test_copy_ops():
    """
    Test the copy operations between tensorflow variables
    """
    tf.reset_default_graph()

    with tf.Session() as sess:

        with tf.variable_scope('online'):
            online_params = make_vars(4)

        with tf.variable_scope('target'):
            target_params = make_vars(4)

        copy_ops, tau = make_copy_ops(online_params, target_params)
        sess.run(tf.global_variables_initializer())

        online_vals, target_vals = sess.run(
            [online_params, target_params]
        )

        assert np.sum(online_vals) != np.sum(target_vals)

        _  = sess.run(copy_ops, {tau: 1.0})

        online_vals, target_vals = sess.run(
            [online_params, target_params]
        )
        assert np.sum(online_vals) == np.sum(target_vals)
