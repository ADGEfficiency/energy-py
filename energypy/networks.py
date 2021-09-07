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
    inputs,
    outputs,
    size_scale=1,
):
    if isinstance(inputs, tuple):
        mask = keras.Input(shape=(inputs[0], inputs[0]))
        inputs = keras.Input(shape=inputs)
    else:
        #  inputs already tensor
        mask = keras.Input(shape=(inputs.shape[1], inputs.shape[1]))

    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs, attention_mask=mask)
    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(net, net)
    net = layers.Flatten()(net)
    outputs = layers.Dense(outputs, activation="linear")(net)
    return [inputs, mask], outputs

