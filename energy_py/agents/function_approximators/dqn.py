import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class tfValueFunction(object):
    """
    A TensorFlow value function approximating either V(s) or Q(s,a).

    The value function approximates the expected future discounted reward.

    V(s) can be modelled by
        input_nodes = observation_space.shape[0]
        output_nodes = 1
        action_index = 0 always

    Q(s,a) can be modelled by
        input_nodes = observation_space.shape[0]
        output_nodes = action_space.shape[0]

    args
        model_dict (dict):
    """
    def __init__(self, model_dict, scope='value_function'):
        logger.info('creating {}'.format(scope))

        for k, v in model_dict.items():
            logger.info('{} : {}'.format(k, v))

        #  network structure
        self.input_nodes = model_dict['input_nodes']
        self.output_nodes = model_dict['output_nodes']
        self.layers = model_dict['layers']

        #  optimizer parameters
        self.lr = model_dict['lr']

        self.scope = scope
        with tf.variable_scope(self.scope):
            #  create graph for prediction and learning
            self.prediction = self.make_prediction_graph()

            self.train_op = self.make_learning_graph()

    def make_prediction_graph(self):
        """
        Creates the prediction part of the graph
        """
        self.obs = tf.placeholder(tf.float32,
                                  [None, self.input_nodes],
                                  'observation')

        #  add the input layer
        with tf.variable_scope('input_layer'):
            layer = tf.layers.dense(self.obs, 
                                    units=self.layers[0],
                                    activation=tf.nn.relu)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[1:]):
            with tf.variable_scope('input_layer_{}'.format(i)):
                layer = tf.layers.dense(inputs=layer,
                                        units=nodes,
                                        activation=tf.nn.relu)

        #  return the prediction
        with tf.variable_scope('output_layer'):
            wt_init = tf.random_uniform_initializer(minval=-0.003,
                                                    maxval=0.003)
            prediction = tf.layers.dense(inputs=layer,
                                    units=self.output_nodes,
                                         kernel_initializer=wt_init)
            prediction = tf.reshape(prediction, [-1, self.output_nodes])
        return prediction

    def make_learning_graph(self):
        """
        Part of the graph used to improve the value function

        Minimizing the squared difference between the prediction and target
        """
        self.target = tf.placeholder(tf.float32,
                                     [None],
                                     'target')

        self.action_index = tf.placeholder(tf.int32,
                                         [None],
                                         'act_indicies')

        rng = tf.range(tf.shape(self.prediction)[0])
        self.Q_act = tf.gather_nd(self.prediction,
                                  tf.stack((rng, self.action_index), -1))

        #  error is calculted explcitly so we can use it outside the
        #  value function - ie as the TD error signal in Actor-Critic
        self.error = self.target - self.Q_act

        #  use the Huber loss as the cost function
        #  we use this to clip gradients
        #  the shape of the huber loss means the slope is clipped at 1
        #  for large errors
        self.loss = tf.losses.huber_loss(self.target, self.Q_act)

        #  create the optimizer object and the training operation
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        return self.train_op

    def predict(self, sess, obs):
        """
        args
            sess (tf.Session) : the current TensorFlow session
            obs (np.array) : observation of the environment
                             shape=(1, self.input_dim)
        """
        return sess.run(self.prediction, {self.obs: obs})

    def improve(self, sess, obs, target, action_index):
        """
        Improving the value function approximation

        either V(s) or Q(s,a) for all a

        The target is created externally to this object
        Most commonly the target will be a Bellman approximation
        V(s) = r + yV(s')
        Q(s,a) = r + yQ(s',a)

        args
            sess (tf.Session) : the current TensorFlow session
            obs (np.array) : shape=(num_samples, self.input_dim)
            target (np.array) : shape=(num_samples, self.output_nodes)
            action_index (int): shape=(num_samples, 1)
        """
        feed_dict = {self.obs: obs,
                     self.target: target,
                     self.action_index: action_index}
        _, error, loss = sess.run([self.train_op, self.error, self.loss],
                                  feed_dict=feed_dict)
        return error, loss

    def save_model(self):
        pass

    def load_model(self):
        pass

    def copy_weights(self, sess, parent):
        self_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(self.scope)]
        self_params = sorted(self_params, key=lambda v: v.name)

        parent_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(parent.scope)]
        parent_params = sorted(parent_params, key=lambda v: v.name)

        update_ops = []
        for parent_p, self_p in zip(parent_params, self_params):
            op = parent_p.assign(self_p)
            update_ops.append(op)

        sess.run(update_ops)

        return sess
