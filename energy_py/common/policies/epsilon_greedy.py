import tensorflow as tf


def epsilon_greedy_policy(q_values,
                          discrete_actions,
                          step_tensor,
                          decay_steps,
                          initial_epsilon,
                          final_epsilon):
    """
    epsilon greedy action selection

    args
        q_values (tensor) (batch_size, num_actions)
        discrete_actions (tensor) (num_actions, *action_shape)
        step_tensor (tensor)
        initial_epsilon (float)
        final_epsilon (float)
        epsilon_decay_fraction (float)

    returns
        epsilon (tensor)
        actions (tensor) (batch_size, action_dimensions)

    With some probability epsilon, either:
        random action is selected
        optimal action selected
    """
    batch_size = tf.shape(q_values)[0]
    num_actions = tf.cast(tf.shape(q_values)[1], tf.int64)

    greedy_action_indicies = tf.argmax(q_values, axis=1)

    #  generating a random index from 0 -> num_actions
    random_action_indicies = tf.random_uniform(
        tf.stack([batch_size]),
        minval=0,
        maxval=num_actions,
        dtype=tf.int64)

    #  generating a probability, one for each sample
    probabilities = tf.random_uniform(
        tf.stack([batch_size]),
        minval=0,
        maxval=1,
        dtype=tf.float32)

    epsilon = tf.train.polynomial_decay(
                learning_rate=initial_epsilon,
                global_step=step_tensor,
                decay_steps=decay_steps,
                end_learning_rate=final_epsilon,
                power=1.0,
                name='epsilon')

    select_greedy = tf.squeeze(tf.greater(probabilities, epsilon))

    indicies = tf.where(
        select_greedy,
        greedy_action_indicies,
        random_action_indicies)

    return epsilon, tf.gather(discrete_actions, indicies)
