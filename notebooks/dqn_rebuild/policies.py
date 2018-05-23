"""
Reinforcement learning policies - functions that map state -> action

Different policies can accept different args but should always return an action
"""

import numpy as np
import tensorflow as tf


def e_greedy(q_values, epsilon, discrete_actions):
    """
    epsilon greedy action selection

    args
        q_values (tensor) (batch_size, num_actions)
        epsilon (tensor) (1,)
        discrete_actions_list (list) usually a list of numpy arrays

    returns
        actions (tensor) (batch_size, action_dimensions)

    With some probability epsilon, either:
        random action is selected
        optimal action selected
    """
    batch_size = tf.shape(q_values)[0]
    num_actions = tf.cast(tf.shape(q_values)[1], tf.int64)

    assert discrete_actions.ndim == 2
    discrete_actions = tf.Variable(discrete_actions, 'discrete_actions')

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

    select_greedy = tf.squeeze(tf.greater(probabilities, epsilon))

    indicies = tf.where(
        select_greedy,
        greedy_action_indicies,
        random_action_indicies)

    return tf.gather(discrete_actions, indicies)


def test_e_greedy_policy():

    #  check that epsilon at zero gives us the best actions
    optimals = sess.run(e_g,
                 {q_values: test_q_values,
                  epsilon: 0.0})

    assert optimals[0].all() == discrete_actions[1].all()
    assert optimals[1].all() == discrete_actions[2].all()
    assert optimals[2].all() == discrete_actions[0].all()

    #  check that epislon at one gives random actions
    randoms = sess.run(e_g,
                 {q_values: test_q_values,
                  epsilon: 1.0})

    one_different = False

    for opt, random in zip (optimals, randoms):
        if opt.all() == random.all():
            pass
        else:
            one_different = True


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

    return temperature, log_probs, entropy, samples


if __name__ == '__main__':


    #  setup for testing
    num_samples = 5
    num_actions = 3
    act_dims = 4

    test_q_values = np.zeros(num_samples * num_actions).reshape(num_samples, -1)
    test_q_values[0, 1] = 1
    test_q_values[1, 2] = 1
    test_q_values[2, 0] = 1

    discrete_actions = np.array(
        [np.random.uniform(size=act_dims)
         for _ in range(num_samples)]).reshape(num_samples, -1)

    #  placeholders for testing
    q_values = tf.placeholder(shape=(None, num_actions), dtype=tf.float32)
    epsilon = tf.placeholder(shape=(), dtype=tf.float32)

    #  construct the tf graph for testing
    e_g = e_greedy(q_values, epsilon, discrete_actions)

    temperature = tf.placeholder(shape=(),
                                 dtype=tf.float32,
                                 name='softmax_temperature')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_e_greedy_policy()

        temp, probs, entropy, samples = softmax_policy(q_values, temperature)

        t, p, e, s = sess.run([temp, probs, entropy, samples],
                              {q_values: test_q_values,
                               temperature: 0.000005})
    optimals = s
    assert optimals[0].all() == discrete_actions[1].all()
    assert optimals[1].all() == discrete_actions[2].all()
    assert optimals[2].all() == discrete_actions[0].all()
    print(t, p, e, s)
