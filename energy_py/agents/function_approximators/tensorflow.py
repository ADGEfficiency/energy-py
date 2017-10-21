import numpy as np
import tensorflow as tf

from energy_py.main.scripts.tensorflow_machinery import fc_layer


def fc_layer(input_tensor, wt_shape, bias_shape, activation=[]):
    """
    Creates a single layer of a feedforward neural network in TensorFlow.

    args
        input_tensor : Tensor (1, input_length)
        wt_shape     :
        bias_shape   :
        activation   : TensorFlow activation function

    return
        output      : output layer of the feedforward neural network
    """
    #
    wt_init = tf.random_uniform_initializer(minval=-0.003, maxval=-0.003)
    #  set min bias to 0.01 to get all relus to fire
    bias_init = tf.constant_initializer(0.01)

    W = tf.get_variable('W', wt_shape, initializer=wt_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    matmul = tf.matmul(input_tensor, W) + b

    if activation:
        output = activation(matmul)
    else:
        output = matmul
    return output


class TF_FunctionApproximator(object):
    """
    The base class for TensorFlow energy_py function approximators.
    """
    def __init__(self, action_space,
                       observation_space):

        self.action_space = kwargs.pop('action_space')
        self.observation_space = kwargs.pop('observation_space')

        self.num_actions =  len(self.action_space)
        self.observation_dim = len(self.observation_space)


class TF_GaussianPolicy(TF_FunctionApproximator):
    """
    A Gaussian policy approximated using a TensorFlow neural network.

    Mean and standard deviation are parameterized for each action.

    args
        action_space        : list of energy_py Space objects
        observation_space   : list of energy_py Space objects
        learning rate       : scalar
    """
    def __init__(self, **kwargs):

        self.learning_rate = kwargs.pop('learning_rate')

        super().__init__(**kwargs)

        self.action = self.make_acting_graph()
        self.train_step = self.make_learning_graph()

    def make_acting_graph(self):
        #  the TensorFlow graph for the policy network
        output_dim = 2 * self.num_actions  # one for mean, one for standard deviation

        #  initialize the TensorFlow machinery
        with tf.name_scope('policy_network'):
            #  create placeholder variable for the observation
            self.observation = tf.placeholder(tf.float32,
                                              [None, self.observation_dim],
                                              'observation')

        with tf.variable_scope('action_selection'):
            #  make a three layer fully-connected neural network
            with tf.variable_scope('input_layer'):
                input_layer = fc_layer(self.observation, [self.observation_dim, self.observation_dim], [self.observation_dim], tf.nn.relu)

            with tf.variable_scope('hidden_layer_1'):
                hidden_layer_1 = fc_layer(input_layer, [self.observation_dim, self.observation_dim * 2], [self.observation_dim*2], tf.nn.relu)

            with tf.variable_scope('hidden_layer_2'):
                hidden_layer_2 = fc_layer(hidden_layer_1, [self.observation_dim*2, self.observation_dim*2], [self.observation_dim*2], tf.nn.relu)

            with tf.variable_scope('output_layer'):
                self.output_layer = fc_layer(hidden_layer_2, [self.observation_dim*2, output_dim], [output_dim])

            #  parameterizing normal distributions
            #  indexes for the output layer
            mean_idx = tf.range(start=0, limit=output_dim, delta=2)
            stdev_idx = tf.range(start=1, limit=output_dim, delta=2)

            #  gather ops
            means = tf.gather(params=self.output_layer, indices=mean_idx, axis=1)
            stdevs = tf.gather(params=self.output_layer, indices=stdev_idx, axis=1)

            #  clip the stdev so that stdev is not zero
            #  TODO not sure what the minimum bound for this should be
            stdevs = tf.clip_by_value(stdevs, 0.1, tf.reduce_max(stdevs))
            self.norm_dist = tf.contrib.distributions.Normal(loc=means, scale=stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample(1)

            #  clipping the action
            lows = np.array([space.low for space in self.action_space])
            highs = np.array([space.high for space in self.action_space])
            self.action = tf.clip_by_value(self.action, lows, highs)

        return self.action

    def make_learning_graph(self):
        with tf.variable_scope('learning'):
            self.taken_action = tf.placeholder(tf.float32, [None, self.num_actions], name='taken_actions')
            self.discounted_return = tf.placeholder(tf.float32, [None, 1], 'discounted_returns')

            self.probs = self.norm_dist.prob(self.taken_action)
            self.probs_clipped = tf.clip_by_value(self.probs, 1e-10, 1)
            self.log_probs = tf.log(self.probs_clipped)

            #  we make use of the fact that multiply broadcasts here
            #  discounted returns is of shape (samples, 1)
            #  while log_probs is of shape (samples, num_actions)
            loss = tf.multiply(self.log_probs, -self.discounted_return)
            self.loss = tf.sum(loss)

            #  creating the training step
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

        return self.train_step

    def get_action(self, session, observation):
        #  generating an action from the policy network
        action = session.run(self.action, {self.observation : observation})
        action = action.reshape(self.num_actions)
        return action

    def improve(self, session,
                             observations,
                             actions,
                             discounted_returns):

        feed_dict = {self.observation : observations,
                     self.taken_action : actions,
                     self.discounted_return : discounted_returns}

        _, loss = session.run([self.train_step, self.loss], feed_dict)

        return loss


class TF_ValueFunction(TF_FunctionApproximator):
    """
    The class for a TensoFlow value function V(s).

    The value function approximates the expected discounted reward
    after leaving state s.

    args
        action_space        : list of energy_py Space objects
        observation_space   : list of energy_py Space objects
    """
    def __init__(self, **kwargs):

        self.layers = kwargs.pop('layers')

        super().__init__(**kwargs)

        self.prediction = make_prediction_graph()

    def make_prediction_graph(self):
        with tf.variable_scope('prediction'):
            with tf.variable_scope('input_layer'):
                layer = fc_layer(self.observation, [self.observation_dim, layers[0]], [layers[0]], tf.nn.relu)

            with tf.variable_scope('hidden_layers'):
                for i, nodes in enumerate(self.layers[:-1]):
                    layer = fc_layer(layer, [nodes, self.layers[i+1]], [self.layers[i+1]], tf.nn.relu)

            with tf.variable_scope('output_layer'):
                self.prediction = fc_layer(layer, [self.layers[i+1], 1], [1])

        return self.prediction

    def predict(self, state):

        return

    def improve(self, states,
                      targets):

        return
