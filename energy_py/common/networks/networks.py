import tensorflow as tf

from energy_py.common.networks.layers import fully_connected_layer


def feed_forward(
        scope,
        input_tensor,
        input_shape,
        hiddens,
        output_nodes,
        output_activation='linear'
):
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
            activation=output_activation
        )

    return output_layer
