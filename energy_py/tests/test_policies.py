
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

