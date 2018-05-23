"""
Functions to generate tensorflow neural network components
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers


def fully_connected_layer(input_tensor, 
                          nodes, 
                          scope, 
                          activation,
                          layer_norm=False):
    """
    A single layer

    Weights are initialized using Xavier-Glorot, biases at zero (these are the defaults )

    Activation function is done using a conditional to allow the layer norm before
    the activation

    """
    layer = layers.fully_connected(
        input_tensor,
        num_outputs=nodes,
        activation_fn=None,
        scope='{}_layer'.format(scope))

    if layer_norm:
        layer = layers.layer_norm(
            layer,
            center=True,
            scale=True,
            scope='{}_layer_norm'.format(scope))

    if activation == 'relu':
        layer = tf.nn.relu(
            layer,
            name='{}_activation'.format(scope))

    elif activation == 'linear':
        pass

    else:
        raise ValueError('Activation of {} not supported'.format(activation))

    return layer


def feed_forward_network(input_tensor,
                         hiddens,
                         num_outputs,
                         scope,
                         hidden_activation='relu',
                         output_activation='linear',
                         layer_norm=False,
                         reuse=False):
    """
    Multiple layers - aka multilayer perceptron

    """
    with tf.variable_scope(scope, reuse=reuse):
        layer = input_tensor

        for num, hidden in enumerate(hiddens):

            layer = fully_connected_layer(
                layer,
                hidden,
                'hidden_{}'.format(num),
                activation=hidden_activation)

        return fully_connected_layer(
            layer,
            num_outputs,
            'output_layer',
            activation=output_activation)
