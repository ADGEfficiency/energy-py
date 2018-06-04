import tensorflow as tf


def softmax_policy(q_values, temperature):
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
    softmax = tf.nn.softmax(tf.divide(q_values, temperature), axis=1)

    log_probs = tf.log(softmax)

    entropy = -tf.reduce_sum(softmax * log_probs, 1, name='softmax_entropy')

    samples = tf.multinomial(log_probs, num_samples=1)

    return softmax, log_probs, entropy, samples
