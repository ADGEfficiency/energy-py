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

from new_dqn import DQN


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


















def test_variable_sharing():
    """
    Tests the variable sharing for DDQN
    """
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent, batch, env = setup_agent(sess, double_q=True)

        obs = batch['observation']
        next_obs = batch['next_observation']

        #  test that our weights are being shared correctly
        online_vals = sess.run(
            agent.online_q_values,
            {agent.observation: next_obs}
        )

        online_copy_vals = sess.run(
            agent.online_next_obs_q,
            {agent.next_observation: next_obs}
        )
        assert np.sum(online_copy_vals) == np.sum(online_vals)
        assert online_copy_vals.all() == online_vals.all()

        #  run again using the observations, to get different results
        online_vals_ = sess.run(
            agent.online_q_values,
            {agent.observation: obs}
        )
        assert np.sum(online_vals) != np.sum(online_vals_)

        #  do a training operation and check again
        agent.learn()

        online_vals = sess.run(
            agent.online_q_values,
            {agent.observation: next_obs}
        )

        online_copy_vals = sess.run(
            agent.online_next_obs_q,
            {agent.next_observation: next_obs}
        )
        assert np.sum(online_copy_vals) == np.sum(online_vals)
        assert online_copy_vals.all() == online_vals.all()


def test_dqn_copy_ops():
    """
    Testing the online & target net copy operations
    """
    #  initialize agent, should be the same

    #  learn, should be different

    #  copy using tau=1.0

    #  copy using tau=0.5
    pass




if __name__ == '__main__':
    test_action_selection()
    test_bellman_target()
    test_ddqn_bellman()
    test_variable_sharing()
