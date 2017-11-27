import numpy as np
import tensorflow as tf


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
    wt_init = tf.constant_initializer(1)
    bias_init = tf.constant_initializer(0)

    W = tf.get_variable('W', wt_shape, initializer=wt_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    matmul = tf.matmul(input_tensor, W) + b

    if activation:
        output = activation(matmul)
    else:
        output = matmul
    return output

class GaussianPolicy(object):
    """
    A Gaussian policy approximated using a TensorFlow neural network.

    Mean and standard deviation are parameterized for each action.

    args
        observation_dim (int)
        num_actions (int)
        learning rate (float)
    """
    def __init__(self, **kwargs):

        self.lr = kwargs.pop('lr')

        self.observation_dim = kwargs.pop('observation_dim')
        self.num_actions = kwargs.pop('num_actions')
        self.action_space = kwargs.pop('action_space')
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
                input_layer = fc_layer(self.observation, [self.observation_dim, 4], [4], tf.nn.relu)

            # with tf.variable_scope('hidden_layer_1'):
            #     hidden_layer_1 = fc_layer(input_layer, [4, 4], [4], tf.nn.relu)

            # with tf.variable_scope('hidden_layer_2'):
            #     hidden_layer_2 = fc_layer(hidden_layer_1, [4, 4], [4], tf.nn.relu)

            with tf.variable_scope('output_layer'):
                self.output_layer = fc_layer(input_layer, [4,  output_dim], [output_dim])

            #  parameterizing normal distributions
            #  one mean & standard deviation per action
            #  as per TRPO paper we parameterize log(standard deviation)
            #  see Schulman et. al (2017) Trust Region Policy Optimization

            #  indexes for the output layer
            mean_idx = tf.range(start=0, limit=output_dim, delta=2)
            stdev_idx = tf.range(start=1, limit=output_dim, delta=2)

            #  gather ops
            self.means = tf.gather(params=self.output_layer, indices=mean_idx, axis=1)
            self.stdevs = tf.gather(params=self.output_layer, indices=stdev_idx, axis=1)
            #self.stdevs = tf.nn.softplus(stdevs) + 1e-5
            self.norm_dist = tf.contrib.distributions.Normal(loc=self.means, scale=self.stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample()

            #  clipping the action
            #  use an exception to catch the single action case
            try:
                lows = np.array([space.low for space in self.action_space])
                highs = np.array([space.high for space in self.action_space])
            except TypeError:
                lows = np.array(self.action_space.low)
                highs = np.array(self.action_space.high)

            self.action = tf.clip_by_value(self.action, lows, highs)

        return self.action

    def make_learning_graph(self):
        with tf.variable_scope('learning'):
            self.taken_action = tf.placeholder(tf.float32, [None, self.num_actions],'taken_actions')
            self.returns = tf.placeholder(tf.float32, [None, 1], 'discounted_returns')

            self.log_probs = self.norm_dist.log_prob(self.taken_action)

            #  we make use of the fact that multiply broadcasts here
            #  discounted returns is of shape (samples, 1)
            #  while log_probs is of shape (samples, num_actions)
            pg_loss = tf.reduce_mean(-self.log_probs * self.returns)

            #  add in some cross entropy cost for exploration
            ce_loss = tf.reduce_mean(1e-1 * self.norm_dist.entropy())
            self.loss = pg_loss - ce_loss

            #  creating the training step
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.loss)

        return self.train_step

    def get_action(self, session, observation):
        assert observation.shape[0] == 1

        #  generating an action from the policy network
        results = session.run([self.means, self.stdevs, self.action], {self.observation : observation})

        output = {'means' : results[0],
                  'stdevs': results[1],
                  'action': results[2]}
        print(output)
        return output['action'].reshape(self.num_actions), output

    def improve(self, session,
                      observations,
                      actions,
                      returns):

        assert observations.shape[0] == actions.shape[0]
        assert actions.shape[0] == returns.shape[0]

        feed_dict = {self.observation : observations,
                     self.taken_action : actions,
                     self.returns : returns}

        _, loss = session.run([self.train_step, self.loss], feed_dict)

        return float(loss)


class TensorflowV(object):
    """
    The class for a TensorFlow value function approximating V(s).

    The value function approximates the expected discounted reward
    after leaving state s.

    args
        observation_space (list) : list of energy_py space objects
        lr (float) : learning rate
        layers (list) : list of nodes per layer 
    """
    def __init__(self, **kwargs):

        self.observation_space = kwargs.pop('observation_space')
        self.lr = kwargs.pop('lr')
        self.layers = kwargs.pop('layers')
        super().__init__(**kwargs)

        self.observation_dim = self.observation_space.shape[0]

        #  create the tf graph for prediction and learning
        self.make_prediction_graph()
        self.make_learning_graph()

    def make_prediction_graph(self):
        """
        Creates the prediction part of the graph
        Predicting V(s) for the observed state
        """
        with tf.variable_scope('prediction'):
            #  create placeholder variable for the observation
            self.obs = tf.placeholder(tf.float32,
                                              [None, self.observation_dim],
                                              'observation')

            with tf.variable_scope('input_layer'):
                layer = fc_layer(self.obs, 
                                 [self.observation_dim, self.layers[0]], 
                                 [self.layers[0]], 
                                 tf.nn.relu)

            with tf.variable_scope('hidden_layers'):
                for i, nodes in enumerate(self.layers[:-1]):
                    layer = fc_layer(layer, 
                                     [nodes, self.layers[i+1]], 
                                     [self.layers[i+1]], 
                                     tf.nn.relu)

            with tf.variable_scope('output_layer'):
                #  single output node to predict V(s)
                self.prediction = fc_layer(layer, 
                                           [self.layers[i+1], 1],
                                           [1])

    def make_learning_graph(self):
        """
        Part of the graph used to improve the value function

        Minimizing the squared difference between the prediction and target
        """
        with tf.variable_scope('learning'):
            #  placeholder for the target
            self.target = tf.placeholder(tf.float32, [None, 1], 'target')
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            
            self.error = tf.subtract(self.prediction, self.target)
            self.loss = tf.square(self.error)
            self.train_step = self.optimizer.minimize(self.loss)

    def predict(self, sess, observation):
        """
        Predicting V(s) for the given observation

        args
            sess (tf.Session) : the current TensorFlow session
            observation (np.array) : 
        """
        return sess.run(self.prediction, {self.obs: observation})

    def improve(self, sess, observation, target):
        """
        Improving V(s) for the given observation
        The target is created externally to this object 
        Most commonly the target will be a Bellman approximation
        V(s) = r + yV(s')

        args
            sess (tf.Session) : the current TensorFlow session
            observation (np.array) : 
            target (np.float) : 
        """
        fd = {self.obs: observation, self.target: target}
        _, error, loss = sess.run([self.train_step, self.error, self.loss], 
                                  feed_dict=fd)
        return error, loss


class TensorflowQ(object):
    """
    The class for a TensorFlow action-value function approximating Q(s,a)

    The action-value function approximates the expected discounted reward
    after taking action a in state s.

    args
        observation_space (list) : list of energy_py space objects
        action_space (list) : list of energy_py space objects
        lr (float) : learning rate
        layers (list) : list of nodes per layer 
    """
    def __init__(self, **kwargs):

        self.observation_space = kwargs.pop('observation_space')
        self.action_space = kwargs.pop('action_space')
        self.lr = kwargs.pop('lr')
        self.layers = kwargs.pop('layers')
        super().__init__(**kwargs)

        self.obs_act_dim = len(self.observation_space) + len(self.action_space)

        #  create the tf graph for prediction and learning
        self.prediction = make_prediction_graph()
        self.train_step = self.make_learning_graph()


    def make_prediction_graph(self):
        """
        Creates the prediction part of the graph
        Predicting V(s) for the observed state

        pretty much identical tto V(s) - probably can rewrite with single function
        """
        with tf.variable_scope('prediction'):
            #  create placeholder variable for the observation
            self.obs_act = tf.placeholder(tf.float32,
                                              [None, self.obs_act_dim],
                                              'observation_action')

            with tf.variable_scope('input_layer'):
                layer = fc_layer(self.obs_act, 
                                 [self.obs_act_dim, layers[0]], 
                                 [layers[0]], 
                                 tf.nn.relu)

            with tf.variable_scope('hidden_layers'):
                for i, nodes in enumerate(self.layers[:-1]):
                    layer = fc_layer(layer, 
                                     [nodes, self.layers[i+1]], 
                                     [self.layers[i+1]], 
                                     tf.nn.relu)

            with tf.variable_scope('output_layer'):
                #  single output node to predict Q(s,a)
                self.prediction = fc_layer(layer, 
                                           [self.layers[i+1], 1],
                                           [1])

    def make_learning_graph(self):
        """
        Part of the graph used to improve the value function

        Minimizing the squared difference between the prediction and target
        """
        with tf.variable_scope('learning'):
            #  placeholder for the target
            self.target = tf.placeholder(tf.float32, [none, 1], 'target')
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            
            self.error = tf.subtract(self.prediction, target)
            self.loss = tf.square(self.error)
            self.train_step = self.optimizer.minimize(self.loss)

    def predict(self, sess, observation_action):
        """
        Predicting Q(s,a) for the given observation

        args
            sess (tf.Session) : the current TensorFlow session
            observation_action (np.array) : 
        """
        return sess.run(self.prediction, {self.obs_act: observation_action})

    def improve(self, sess, observation_action, target):
        """
        Improving V(s) for the given observation
        The target is created externally to this object 
        Most commonly the target will be a Bellman approximation
        V(s) = r + yV(s')

        args
            sess (tf.Session) : the current TensorFlow session
            observation (np.array) : 
            target (np.float) : 
        """
        fd = {self.obs_act: observation_action, self.target: target}
        _, error, loss = sess.run([self.train_step, self.error, self.loss], 
                                  feed_dict=fd)
        return error, loss
