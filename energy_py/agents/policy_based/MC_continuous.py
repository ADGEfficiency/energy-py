def gaussian_policy_network(obs, obs_dim, num_actions):
    """
    Builds a TensorFlow graph with two outputs per action.
    Parameterizes both the mean & standard deviation of a Normal distributon.

    Ref = [1]
    """
    action_dim = 2 * num_actions  # one for mean, one for standard deviation

    with tf.variable_scope('input_layer'):
        input_layer = fc_layer(obs, [obs_dim, 50], [50], tf.nn.relu)

    with tf.variable_scope('hidden_layer'):
        hidden_layer = fc_layer(input_layer, [50, 100], [100], tf.nn.relu)

    with tf.variable_scope('output_layer'):
        output_layer = fc_layer(hidden_layer, [100, 2 * action_dim], [2 * action_dim])

    mean = tf.gather(output_layer, 0)
    stdev = tf.gather(output_layer, 1)
    stdev = ttf.nn.softplus(stdev) + 1e-5
    norm_dist = tf.contrib.distributions.Normal(mean, stdev)
    return action

class PolicyGradientAgent(Base_Agent):
    """
    Monte-Carlo policy gradient agent for continuous action spaces.
    """
    def __init__(self, obs_dim,
                       num_actions):

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.memory = Agent_Memory()

    def make_graph(self):
        """
        Makes a TensorFlow graph for a Gaussian Policy Network
        """
        with tf.variable_scope('policy_network'):
            #  placeholder variables
            self.observation = tf.placeholder(tf.float32, obs_dim, 'observation')
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), 'discounted_rewards')

            #  the TensorFlow graph for the policy network
            normal_dist = gaussian_policy_network(self.observation, self.obs_dim, self.num_actions)
            self.action = normal_dist._sample_n(1)

            self.loss = - normal_dist.log_prob(self.action) * discounted_rewards
            self.optimizer = tf.train.AdamOptimizer()
            self.train_step = self.optimizer.minimize(loss)
        return None

    def _select_action(session, obs):
        return session.run(action, {self.observation:obs})

    def _learn(session, memory):
        feed_dict = {self.observation : obs,
                     self.action : action,
                     self.discounted_rewards : discounted_rewards}

        _, loss = session.run([self.train_step, self.loss], feed_dict)
        return loss
