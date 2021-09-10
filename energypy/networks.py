import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dense(
    input_shape,
    outputs,
    size_scale=1,
):
    if isinstance(input_shape, tuple):
        input_shape = keras.Input(shape=input_shape)

    if isinstance(input_shape, list):
        input_shape = tf.concat(input_shape, axis=1)

    net = layers.Dense(64 * size_scale, activation="relu")(input_shape)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    outputs = layers.Dense(outputs, activation="linear")(net)
    return input_shape, outputs


def attention(
    input_shape,
    outputs,
    size_scale=1,
):
    if isinstance(input_shape, tuple):
        mask = keras.Input(shape=(input_shape[0], input_shape[0]))
        inputs = keras.Input(shape=input_shape)
    else:
        #  input_shape already a tensor
        mask = keras.Input(shape=(input_shape.shape[1], input_shape.shape[1]))

    net = layers.MultiHeadAttention(num_heads=4, key_dim=32 * size_scale)(
        inputs, inputs, attention_mask=mask
    )
    net = layers.MultiHeadAttention(num_heads=4, key_dim=32 * size_scale)(net, net)
    net = layers.Flatten()(net)
    outputs = layers.Dense(outputs, activation="linear")(net)
    return [inputs, mask], outputs
