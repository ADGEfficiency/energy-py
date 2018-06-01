"""
Functions to generate tensorflow neural network components
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import energy_py


def fully_connected_layer(scope,
                          input_tensor,
                          input_shape,
                          output_nodes,
                          activation='relu'):
    """
    Creates a single fully connected layer

    args
        scope (str) usually 'input_layer' or 'hidden_layer_2' etc
        input_tensor (tensor)
        input_shape (tuple or int)
        output_nodes (int)
        activation (str) currently support relu or linear

    To correctly name the variables and still allow variable sharing:
    with tf.name_scope('online_network):
        layer = fully_connected_layer('input_layer', ...)

    """
    #  feed input shape as a tuple for support for high dimensional inputs
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'weights',
            shape=(*input_shape, output_nodes),
            initializer=tf.contrib.layers.xavier_initializer()
        )

        bias = tf.get_variable(
            'bias',
            shape=(output_nodes),
            initializer=tf.zeros_initializer()
        )

        layer = tf.add(
            tf.matmul(input_tensor, weights),
            bias,
            name='layer'
        )

    if activation == 'relu':
        return tf.nn.relu(layer)

    elif activation == 'linear':
        return layer

    else:
        raise ValueError(
            'Activation of {} not supported'.format(activation))


def feed_forward(scope,
		 input_tensor,
                 input_shape,
                 hiddens,
                 output_nodes):
    """
    Creates a feed forward neural network (aka multilayer perceptron)

    args
	scope (str)
        input_tensor (tensor)
        input_shape (tuple or int)
        hiddens (list) has nodes per layer (includes input layer)
        output_nodes (int)
    """
    with tf.name_scope(scope):
        layer = fully_connected_layer(
            'input_layer',
            input_tensor,
            input_shape,
            hiddens[0])

        for layer_num, nodes in enumerate(hiddens[1:]):
            layer = fully_connected_layer(
                'hidden_layer_{}'.format(layer_num),
                layer,
                (hiddens[layer_num-1],),
                nodes
            )

        output_layer = fully_connected_layer(
            'output_layer',
            layer,
            (hiddens[-1],),
            output_nodes,
            activation='linear'
        )

    return output_layer


def get_tf_params(scope):
    params = [p for p in tf.trainable_variables()
              if p.name.startswith(scope)]

    #  sort parameters list by the variable name
    params = sorted(params, key=lambda var: var.name)

    return params


def make_copy_ops(parent, child, scope='copy_ops'):
    print('making copy ops')
    copy_ops = []

    with tf.variable_scope(scope):
        tau = tf.Variable(1.0, name='tau')

        for p, c in zip(parent, child):
            assert p.name.split('/')[1:] == c.name.split('/')[1:]
            print(p.name, c.name)

            new_value = tf.add(tf.multiply(p, tau),
                               tf.multiply(c, 1 - tau))
            op = c.assign(new_value)
            copy_ops.append(op)

    return copy_ops, tau


def make_vars(num):
    variables = []
    for n in range(num):
        variables.append(tf.get_variable(
            'var_{}'.format(n),
            shape=(6,),
            initializer=tf.initializers.random_normal
        )
                         )
    return variables


if __name__ == '__main__':
    pass
