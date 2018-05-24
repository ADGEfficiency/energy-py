"""
Functions to generate tensorflow neural network components
"""

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


if __name__ == '__main__':
    env = energy_py.make_env('Battery')
    obs = env.observation_space.sample()
    discount = 0.95

    #  import here because of dependencies between this and new_dqn.py
    #  will be sorted eventually
    from new_dqn import DQN
    tf.reset_default_graph()
    with tf.Session() as sess:
        a = DQN(sess=sess, env=env, total_steps=10,
                discount=discount)

        copy_ops, tau = make_copy_ops(a.online_params, a.target_params)

        import numpy as np
        online_vals, target_vals = sess.run(
            [a.online_q_values, a.target_q_values],
            {a.observation: obs,
             a.next_observation: obs}
        )

        #  equal because we intialize target net weights in the init of DQN
        assert np.sum(online_vals) == np.sum(target_vals)


