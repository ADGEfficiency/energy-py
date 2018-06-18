import logging
import tensorflow as tf

from energy_py.common.networks.layers import fully_connected_layer


logger = logging.getLogger(__name__)


def feed_forward(
        scope,
        input_tensor,
        input_shape,
        layers,
        output_nodes,
        output_activation='linear'
):
    """
    Creates a feed forward neural network (aka multilayer perceptron)

    args
        scope (str)
        input_tensor (tensor)
        input_shape (tuple or int)
        layers (list) has nodes per layer (includes input layer)
        output_nodes (int)
    """
    logger.info('Making feed_forward network - {}'.format(scope))
    logger.info('input {} layers {} output {}'.format(
            input_shape, layers, output_nodes))

    with tf.name_scope(scope):
        layer = fully_connected_layer(
            'input_layer',
            input_tensor,
            input_shape,
            layers[0]
        )

        for layer_num, nodes in enumerate(layers[1:]):
            layer = fully_connected_layer(
                'hidden_layer_{}'.format(layer_num),
                layer,
                layers[layer_num-1],
                nodes
            )

        output_layer = fully_connected_layer(
            'output_layer',
            layer,
            layers[-1],
            output_nodes,
            activation=output_activation
        )

    return output_layer
