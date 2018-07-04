import tensorflow as tf


def softmax_policy(
        q_values,
        discrete_actions,
        step_tensor,
        decay_steps,
        initial_temp,
        final_temp
):
    """
    A softmax aka a Boltzmann policy

    args
        q_values (tensor) (batch_size, num_actions)
        temperature (tensor) (1,)

    returns
        actions (tensor) (batch_size, action_dimensions)

    Higher temperature -> more exploration

    Basic structure of the policy from here
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf

    Calculation of entropy from here
    https://github.com/dennybritz/reinforcement-learning/issues/34

    Log of the probabilities are taken for speed, accuracy and stability
    """

    temp = tf.train.polynomial_decay(
                learning_rate=initial_temp,
                global_step=step_tensor,
                decay_steps=decay_steps,
                end_learning_rate=final_temp,
                power=1.0,
                name='temperature'
    )

    softmax = tf.nn.softmax(tf.divide(q_values, temp), axis=1)

    log_probs = tf.log(softmax)

    entropy = -tf.reduce_sum(softmax * log_probs, 1, name='softmax_entropy')

    samples = tf.multinomial(log_probs, num_samples=1)

    policy = tf.gather(discrete_actions, samples)

    return temp, softmax, log_probs, entropy, samples, policy
