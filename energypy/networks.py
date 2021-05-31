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


#  maybe can use the mask to deal with episode progression? TODO
def attention(
    input_shape,
    output_nodes,
    size_scale=1,
):
    inputs = keras.Input(shape=input_shape)
    net, _ = MultiHeadAttention(64 * size_scale, num_heads=8)(inputs)
    net, _ = MultiHeadAttention(32 * size_scale, num_heads=8)(net)
    net = layers.Dense(32 * size_scale, activation="relu")(net)
    outputs = layers.Dense(output_nodes, activation="linear")(net)
    return inputs, outputs


class MultiHeadAttention(tf.keras.layers.Layer):
    """from the TF2 tutorial"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
