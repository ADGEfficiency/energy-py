
import numpy as np
import tensorflow as tf

from energy_py.main.scripts.tensorflow_machinery import fc_layer


class TF_FunctionApproximator(object):
    """
    The base class for TensorFlow energy_py function approximators
    """
    def __init__(self):
        pass


class GaussianPolicy(TF_FunctionApproximator):
    def __init__(self, action_space,
                       observation_space,
                       learning_rate,
                       model_dict):

        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions =  len(self.action_space)
        self.observation_dim = len(self.observation_space)

        self.learning_rate = learning_rate
        self.layers = model_dict['layers']

        self.make_acting_graph()
        self.make_learning_graph()

    def make_acting_graph(self):
        #  the TensorFlow graph for the policy network
        output_dim = 2 * self.num_actions  # one for mean, one for standard deviation

        #  initialize the TensorFlow machinery
        with tf.name_scope('policy_network'):
            #  create placeholder variable for the observation
            self.observation = tf.placeholder(tf.float32,
                                              [None, self.observation_dim],
                                              'observation')

        with tf.variable_scope('prediction'):
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
            #
            stdevs = tf.clip_by_value(stdevs, 1, tf.reduce_max(stdevs))
            self.norm_dist = tf.contrib.distributions.Normal(loc=means, scale=stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample(1)

            #  clipping the action
            lows = np.array([space.low for space in self.action_space])
            highs = np.array([space.high for space in self.action_space])
            self.action = tf.clip_by_value(self.action, lows, highs)

    def make_learning_graph(self):
        #  using the score function to calculate the loss
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
            self.loss = tf.reduce_mean(loss)

            #  creating the training step
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

        return None

    def get_action(self, session, observation):
        #  generating an action from the policy network
        action = session.run(self.action, {self.observation : observation})
        action = action.reshape(self.num_actions)
        return action

    def improve_policy(self, session, observations,
                       actions, discounted_returns):

        feed_dict = {self.observation : observations,
                     self.taken_action : actions,
                     self.discounted_return : discounted_returns}

        _, loss = session.run([self.train_step, self.loss], feed_dict)
        return float(loss)
