"""
Reinforcement learning policies - functions that map state -> action

Different policies can accept different args but should always return an action
"""

import numpy as np
import tensorflow as tf


def e_greedy(q_values,
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


def test_e_greedy_policy():

    #  check that epsilon at zero gives us the best actions
    optimals = sess.run(policy,
                 {q_values: test_q_values,
                  step: decay_steps + 1})

    assert optimals[0].all() == discrete_actions[1].all()
    assert optimals[1].all() == discrete_actions[2].all()
    assert optimals[2].all() == discrete_actions[0].all()

    #  check that epislon at one gives random actions
    randoms = sess.run(policy,
                 {q_values: test_q_values,
                  step: 0})

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
    
    #  TODO should this be a placeholder or a variable???
    #  incrementing step harder in tensorflow than within the agent
    #  also dont want to have to run step in fetches
    #  -> placehoder
    step = tf.placeholder(shape=(), name='learning_step', dtype=tf.int64)

    decay_steps = 10
    epsilon, policy = e_greedy(
        q_values,
        discrete_actions,
        step,
        decay_steps=decay_steps,
        initial_epsilon=1.0,
        final_epsilon=0.0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_e_greedy_policy()

