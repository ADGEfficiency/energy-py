import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class GaussianPolicy(object):
    """
    A Gaussian policy approximated using a TensorFlow neural network.

    Mean and standard deviation are parameterized for each action.

    args
        model_dict (dict)

        todo
            how to set the output nodes vs num actions (in agent or in here?)
    """
    def __init__(self, model_dict, scope='GaussianPolicy'):
        logger.info('creating {}'.format(scope))

        for k, v in model_dict.items():
            logger.info('{} : {}'.format(k, v))

        #  network structure
        self.input_nodes = model_dict['input_nodes']
        self.output_nodes = model_dict['output_nodes']
        self.layers = model_dict['layers']

        #  optimizer parameters
        self.lr = model_dict['lr']

        #  action space so we can clip actions
        self.action_space = model_dict['action_space']

        #  create the TensorFlow graphs
        self.action = self.make_acting_graph()
        self.train_op = self.make_learning_graph()

    def make_acting_graph(self):
        #  the TensorFlow graph for the policy network
        #  initialize the TensorFlow machinery
        with tf.name_scope('policy_network'):
            #  create placeholder variable for the observation
            self.obs = tf.placeholder(tf.float32,
                                              [None, self.input_nodes],
                                              'observation')
            obs = tf.reshape(self.obs, (None, self.input_nodes, 1))

            #  add the input layer
            with tf.variable_scope('inputlayer'):

                layer = tf.layers.dense(inputs=self.obs,
                                        units=self.layers[0],
                                        activation=tf.nn.relu)

            #  iterate over self.layers
            for i, nodes in enumerate(self.layers[1:]):
                with tf.variable_scope('input_layer_{}'.format(i)):
                    layer = tf.layers.dense(inputs=layer,
                                            units=nodes,
                                            activation=tf.nn.relu)

            #  return the means and standard deviations for each action
            with tf.variable_scope('output_layer'):
                self.output_layer = tf.layers.dense(inputs=layer,
                                        units=self.output_nodes)

            #  parameterizing normal distributions
            #  one mean & standard deviation per action
            #  as per TRPO paper we parameterize log(standard deviation)
            #  see Schulman et. al (2017) Trust Region Policy Optimization

            #  indexes for the output layer
            mean_idx = tf.range(start=0, limit=self.output_nodes, delta=2)
            stdev_idx = tf.range(start=1, limit=self.output_nodes, delta=2)

            #  gather ops
            self.means = tf.gather(params=self.output_layer, indices=mean_idx, axis=1)
            stdevs = tf.gather(params=self.output_layer, indices=stdev_idx, axis=1)
            self.stdevs = tf.exp(stdevs) + 1e-5
            self.norm_dist = tf.contrib.distributions.Normal(loc=self.means, scale=self.stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample()

            #  clipping the action
            lows = np.array([space.low for space in self.action_space.spaces])
            highs = np.array([space.high for space in self.action_space.spaces])

            self.action = tf.clip_by_value(self.action, lows, highs)

        return self.action

    def make_learning_graph(self):
        with tf.variable_scope('learning'):
            self.taken_action = tf.placeholder(tf.float32, 
                                               [None, self.output_nodes/2],
                                               'taken_actions')
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
            self.train_op = self.optimizer.minimize(self.loss)

        return self.train_op

    def get_action(self, session, observation):
        assert observation.shape[0] == 1

        #  generating an action from the policy network
        results = session.run([self.means, self.stdevs, self.action], {self.obs : observation})

        output = {'means' : results[0],
                  'stdevs': results[1],
                  'action': results[2]}

        return output['action'].reshape(int(self.output_nodes/2)), output

    def improve(self, 
                session,
                observations,
                actions,
                returns):

        assert observations.shape[0] == actions.shape[0]
        assert actions.shape[0] == returns.shape[0]

        feed_dict = {self.obs : observations,
                     self.taken_action : actions,
                     self.returns : returns}

        _, loss = session.run([self.train_op, self.loss], feed_dict)

        return float(loss)
