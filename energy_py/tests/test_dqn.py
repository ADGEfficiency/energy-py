"""
Test suite for DQN

Issue where I don't ahve any terminals!!!

Maybe better to redo the memory as randomly sampling rather than
stepping
"""
import random

import numpy as np
import tensorflow as tf

import energy_py

from energy_py.scripts.tf_utils import get_tf_params
from energy_py.agents import DQN


def setup_agent(sess, double_q=False):
    """
    Sets up an agent & fills memory

    args
        sess (tf.Session)

    returns
        agent (energy_py DQN agent)
        env (energy_py Battery environment)
    """

    env = energy_py.make_env('Battery')

    #  use high learning rate to get weight changes
    agent = DQN(
        sess=sess,
        env=env,
        total_steps=10,
        discount=0.9,
        memory_type='deque',
        learning_rate=1.0,
        double_q=double_q,
        update_target_net_steps=100,
    )

    for step in range(48):
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        reward = random.random() * 10
        next_obs = env.observation_space.sample()
        done = random.choice([True, False])
        agent.remember(obs, action, reward, next_obs, done)

    batch = agent.memory.get_batch(agent.batch_size)

    return agent, batch, env


def test_action_selection():
    """
    Tests that we select the correct actions
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess)

        #  test of the q_selected action part of the graph
        obs = batch['observation']
        q_vals = sess.run(
            agent.online_q_values,
            {agent.observation: obs}
        )

        selected = np.random.randint(
            low=0,
            high=agent.num_actions,
            size=obs.shape[0]
        ).reshape(-1)

        q_selected = sess.run(
            agent.q_selected_actions,
            {agent.observation: obs,
             agent.selected_action_indicies: selected},
        )

        q_check = q_vals[np.arange(obs.shape[0]), selected]

        np.testing.assert_array_almost_equal(q_check, q_selected)


def test_bellman_target():
    """
    Tests we are forming the Bellman target correctly
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess)

        rewards = batch['reward']
        next_obs = batch['next_observation']
        terms = batch['done']

        #  check the terminal are masking the target network correctly
        #  get the q values for each action for the next observation
        unmasked_next_q = sess.run(
            agent.target_q_values,
            {agent.next_observation: next_obs}
        )

        #  do the argmax
        max_next_q = np.max(unmasked_next_q, axis=1)

        #  this check is mirroring the tensorflow code exactly :D
        check_masked_next_q = np.where(terms.reshape(-1),
                                       max_next_q.reshape(-1),
                                       np.zeros_like(max_next_q))

        masked_next_q = sess.run(
            agent.next_state_max_q,
            {agent.next_observation: next_obs,
             agent.terminal: terms}
        )

        np.testing.assert_equal(
            check_masked_next_q.reshape(-1),
            masked_next_q.reshape(-1)
        )

        #  checking normal q-learning bellman target
        bellman = sess.run(
            agent.bellman,
            {agent.next_observation: next_obs,
             agent.terminal: terms,
             agent.reward: rewards}
        )

        discount = sess.run(agent.discount)

        bellman_check = rewards.reshape(-1) + discount * check_masked_next_q

        np.testing.assert_array_almost_equal(
            bellman_check.reshape(-1),
            bellman.reshape(-1)
        )


def test_ddqn_bellman():
    """
    Checking the DDQN Bellman target creation
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)

        rewards = batch['reward']
        next_obs = batch['next_observation']
        terms = batch['done']

        #  get the optimal next action suggested by the online net
        online_next_obs_q = sess.run(
            agent.online_next_obs_q,
            {agent.next_observation: next_obs}
        )

        online_acts = np.argmax(online_next_obs_q, axis=1)

        #  use this as an integer index on the target net Q(s,a) approximation
        target_net_q_values = sess.run(
            agent.target_q_values,
            {agent.next_observation: next_obs}
        )

        selected_target_next_q = target_net_q_values[np.arange(next_obs.shape[0]),
                                                    online_acts]

        masked = np.where(
            terms.reshape(-1),
            selected_target_next_q,
            np.zeros_like(selected_target_next_q)
        )

        discount = sess.run(agent.discount)
        bellman_check = rewards.reshape(-1) + discount * masked

        bellman = sess.run(
            agent.bellman,
            {agent.next_observation: next_obs,
             agent.terminal: terms,
             agent.reward: rewards}
        )

        np.testing.assert_array_almost_equal(
            bellman_check.reshape(-1),
            bellman.reshape(-1)
        )


def test_target_net_weight_init():
    """
    Testing that the online & target net weights are the same after
    agent is created
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)
        obs = batch['next_observation']

        online_vals, target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )

        #  equal because we intialize target net weights in the init of DQN
        np.testing.assert_array_equal(
            online_vals,
            target_vals
        )

        online_vars = get_tf_params('online')
        target_vars = get_tf_params('target')

        o_vars, t_vars = sess.run([online_vars, target_vars])

        for o_v, t_v in zip(o_vars, t_vars):
            np.testing.assert_array_equal(
                o_v,
                t_v
            )


def test_online_target_initial():
    """
    Testing
    - online & target same before training
    - online difference before & after training
    - target same before and after training
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)
        obs = batch['observation']

        old_online_vals, old_target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )

        np.testing.assert_array_equal(
            old_online_vals,
            old_target_vals
        )

        #  learn - changing the online network weights
        agent.learn()

        online_vals, target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )

        #  check the online and target net values are different
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            online_vals,
            target_vals
        )

        #  check the old and new online values are different
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            old_online_vals,
            online_vals
        )

        #  check the target values haven't changed
        np.testing.assert_array_equal(
            old_target_vals,
            target_vals
        )


def test_variable_sharing():
    """
    Testing
    - online obs & next_obs same before training
    - online obs & next obs same after training
    - online obs & next obs different for different inputs
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)
        obs = batch['observation']
        next_obs = batch['next_observation']

        #  get q values for the two online networks for the
        #  same input
        online_obs, online_next_obs = sess.run(
            [agent.online_q_values,
             agent.online_next_obs_q],
            {agent.observation: obs,
             agent.next_observation: obs}
        )

        np.testing.assert_array_equal(
            online_obs,
            online_next_obs
        )

        #  now check they are different for different inputs
        online_obs, online_next_obs = sess.run(
            [agent.online_q_values,
             agent.online_next_obs_q],
            {agent.observation: obs,
             agent.next_observation: next_obs}
        )

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            online_obs,
            online_next_obs
        )

        #  change weights and check the sharing is still working
        agent.learn()

        online_obs, online_next_obs = sess.run(
            [agent.online_q_values,
             agent.online_next_obs_q],
            {agent.observation: obs,
             agent.next_observation: obs}
        )

        np.testing.assert_array_equal(
            online_obs,
            online_next_obs
        )


def test_copy_ops():
    """
    Testing that different values of tau are working correctly
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)

        #  at this point our target and online networks are the same
        #  (this is tested above in test_online_target_initial)

        #  do a train operation to change the online variables
        agent.learn()

        online_vars = get_tf_params('online')
        target_vars = get_tf_params('target')

        #  get the variable values before we do the copy op
        old_o_vars, old_t_vars = sess.run([online_vars, target_vars])

        #  do the copy operation with tau at 0.5
        _ = sess.run(
            agent.copy_ops,
            {agent.tau: 0.5}
        )

        #  get the new variable values
        new_o_vars, new_t_vars = sess.run([online_vars, target_vars])

        #  check the online variables are the same
        check_o_vars = old_o_vars
        for v1, v2 in zip(check_o_vars, new_o_vars):
            np.testing.assert_array_equal(
                v1,
                v2
            )

        #  calculate what the new target net vars should be
        check_t_vars = []
        for v1, v2 in zip(old_o_vars, old_t_vars):
            new_arr = 0.5 * v1 + 0.5 * v2
            check_t_vars.append(new_arr)

        #  check that the new target vars are what they should be
        for v1, v2 in zip(check_t_vars, new_t_vars):
            np.testing.assert_array_equal(
                v1,
                v2
            )

        #  repeat the same logic with tau = 1
        _ = sess.run(
            agent.copy_ops,
            {agent.tau: 1.0}
        )

        #  get the new variable values
        new_o_vars, new_t_vars = sess.run([online_vars, target_vars])

        #  check that the new target vars are what they should be
        for v1, v2 in zip(new_o_vars, new_t_vars):
            np.testing.assert_array_equal(
                v1,
                v2
            )




if __name__ == '__main__':
    # test_action_selection()
    # test_bellman_target()
    # test_ddqn_bellman()

    # test_target_net_weight_init()
    # test_variable_sharing()

    test_copy_ops()
    test_variable_sharing()
