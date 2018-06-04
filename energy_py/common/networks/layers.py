import tensorflow as tf


def fully_connected_layer(
        scope,
        input_tensor,
        input_shape,
        output_nodes,
        activation='relu'
):
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
