"""
"""

import tensorflow as tf

def fc_layer(input_tensor, wt_shape, bias_shape, activation=None):
    wt_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(0)

    W = tf.get_variable('W', wt_shape, initializer=wt_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    matmul = tf.matmul(input_tensor, W) + b

    if activation:
        output = activation(matmul)
    else:
        output = matmul
    return output
