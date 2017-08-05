import tensorflow as tf

"""
https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_reinforce_cartpole.py
"""


def layer(input_tensor, weight_shape, bias_shape, activation=False):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(0)

    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    matmul = tf.matmul(input_tensor, W) + b

    if activation:
        output = activation(matmul)
    else:
        output = matmul

    return output


def network(data, n_inputs, n_outputs):
    """
    Helper function to create a network.
    """

    with tf.variable_scope('layer_1'):
        layer_1 = layer(observation, [n_inputs, 100], [100], tf.nn.relu)

    with tf.variable_scope('layer_2'):
        layer_2 = layer(layer_1, [100, 100], [100], tf.nn.relu)

    with tf.variable_scope('layer_3'):
        prediction = layer(layer_2, [100, n_outputs], [100])

    return prediction


#  specify network architecture
n_inputs = 4  # len(env.observation_space)
n_outputs = 1 # len(env.action_space)
observation = tf.placeholder(tf.float32, shape=[None, n_inputs])

#  build network
with tf.variable_scope('policy_network'):
    action = network(observation, n_inputs, n_outputs)

with tf.variable_scope('loss'):
    y = tf.placeholder(tf.float32, shape=(None, 1))
    loss = tf.reduce_mean(tf.squared_difference(prediction, y))

with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
