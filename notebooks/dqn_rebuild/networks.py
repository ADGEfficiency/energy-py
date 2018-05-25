"""
Functions to generate tensorflow neural network components
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import energy_py

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
        scope='{}'.format(scope),
    )

    if layer_norm:
        layer = layers.layer_norm(
            layer,
            center=True,
            scale=True,
            scope='{}_layer_norm'.format(scope),
        )

    if activation == 'relu':
        layer = tf.nn.relu(
            layer,
            name='{}_activation'.format(scope))

    elif activation == 'linear':
        pass

    else:
        raise ValueError('Activation of {} not supported'.format(activation))

    return layer


def feed_forward(input_tensor,
                 hiddens,
                 num_outputs,
                 hidden_activation='relu',
                 output_activation='linear',
                 layer_norm=False):
    """
    Multiple layers - aka multilayer perceptron

    """
    layer = input_tensor

    for num, hidden in enumerate(hiddens):

        layer = fully_connected_layer(
            layer,
            hidden,
            'hidden_{}'.format(num),
            activation=hidden_activation,
        )

    output = fully_connected_layer(
        layer,
        num_outputs,
        'output_layer',
        activation=output_activation,
    )

    return output


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
