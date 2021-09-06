import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dense(
    inputs,
    outputs,
    size_scale=1,
):
    if isinstance(inputs, tuple):
        inputs = keras.Input(shape=inputs)

    net = layers.Dense(64 * size_scale, activation="relu")(inputs)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    outputs = layers.Dense(outputs, activation="linear")(net)
    return inputs, outputs


def attention(
    input_shape,
    output_shape,
    size_scale=1,
):
    if isinstance(input_shape, tuple):
        inputs = keras.Input(shape=input_shape)
        mask = keras.Input(shape=(input_shape[0], input_shape[0]))

    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs, attention_mask=mask)
    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(net, net)
    net = layers.Flatten()(net)
    outputs = layers.Dense(output_shape, activation="linear")(net)
    return [inputs, mask], outputs

