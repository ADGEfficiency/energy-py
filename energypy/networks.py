import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten


def dense(input_shape, outputs, size_scale=1, neurons=(64, 32)):
    if isinstance(input_shape, tuple):
        input_shape = keras.Input(shape=input_shape)
        #  think this will break?

    if isinstance(input_shape, list):
        # in_act = input_shape[1]
        # act = tf.expand_dims(in_act, 2)

        in_obs = input_shape[0]
        in_act = input_shape[1]
        obs = Flatten()(in_obs)
        act = Flatten()(in_act)
        inputs = tf.concat([obs, act], axis=1)

    net = Flatten()(inputs)
    for n in neurons:
        net = layers.Dense(n * size_scale, activation="relu")(net)

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
