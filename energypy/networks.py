import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dense(
    inputs,
    output_nodes,
    size_scale=1,
):
    if isinstance(inputs, tuple):
        inputs = keras.Input(shape=inputs)

    net = layers.Dense(64 * size_scale, activation="relu")(inputs)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    outputs = layers.Dense(output_nodes, activation="linear")(net)
    return inputs, outputs


def attention(
    inputs,
    output_nodes,
    size_scale=1,
):
    if isinstance(inputs, tuple):
        inputs = keras.Input(shape=inputs)

    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    net = layers.MultiHeadAttention(num_heads=2, key_dim=32)(net, net)
    net = layers.Flatten()(net)
    outputs = layers.Dense(output_nodes, activation="linear")(net)
    return inputs, outputs

if __name__ == '__main__':
    import numpy as np

    n_features = 3
    sequence_length = 2
    n_samples = 2

    data = np.random.random((n_samples, n_features, sequence_length))

    inp, out = attention((n_features, sequence_length), 1, 1)
    mdl = keras.Model(inputs=inp, outputs=out)
    print(mdl(data))
