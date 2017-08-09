def discrete_action_space_policy_network(obs, obs_dim, num_actions):
    """
    """
    with tf.variable_scope('input_layer'):
        input_layer = fc_layer(obs, [obs_dim, 50], [50], tf.nn.relu)

    with tf.variable_scope('hidden_layer'):
        hidden_layer = fc_layer(input_layer, [50, 100], [100], tf.nn.relu)

    with tf.variable_scope('output_layer'):
        output_layer = fc_layer(hidden_layer)

    return output_layer


class Discrete_PolicyGradientAgent(Base_Agent):
    """
    Monte-Carlo policy gradient agent for discrete action spaces.
    """

    self.output_layer = discrete_action_space_policy_network(obs, self.obs_dim, self.num_actions)
    output_layer =
    action_probs = tf.squeeze(tf.nn.softmax(output_layer))
    taken_action_prob = tf.gather(action_probs, taken_action)

    loss = -tf.log(taken_action_prob) * target

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    def _select_action(session, obs):
        return session.run(action, {self.observation:obs})

    def _learn(session, memory):
        feed_dict = {self.observation : obs,
                     self.action : action,
                     self.discounted_rewards : discounted_rewards}

        _, loss = session.run([self.train_step, self.loss], feed_dict)
        return loss
