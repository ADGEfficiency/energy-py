import logging
import os

import numpy as np
import tensorflow as tf

import energy_py
from energy_py.common.utils import save_args, parse_ini
from energy_py.common.logging import make_logger

from energy_py.experiments import Runner, save_env_info, make_paths, make_config_parser

from energy_py.experiments import process_experiment


def setup_experiment(
        sess,
        agent_config,
        env_config,
        paths,
        seed=None
):
    """

    args
        sess (tf.Session)
        agent_config (dict)
        env_config (dict)
        paths (dict)
        seed (int)
    """

    env = energy_py.make_env(**env_config)
    save_args(env_config, path=paths['env_args'])

    if seed:
        logger.info('random seed is {}'.format(seed))
        env.seed(seed)

    agent_config['env'] = env
    agent_config['sess'] = sess
    agent_config['act_path'] = paths['tb_act']
    agent_config['learn_path'] = paths['tb_learn']

    agent_memory = agent_config.pop('load_memory', None)

    if agent_memory:
        agent_config['load_memory_path'] = paths['memory']

    agent = energy_py.make_agent(**agent_config)
    save_args(agent_config, path=paths['agent_args'])

    if hasattr(agent, 'acting_writer'):
        agent.acting_writer.add_graph(sess.graph)

    return agent, env


def training_experiment(
        sess,
        agent,
        env,
        runner, 
        paths,
        total_steps
):
    """ experiment of multiple episodes with learning """

    logger.info('starting training experiment of {} steps'.format(total_steps))

    #  outer while loop runs through multiple episodes
    step, episode = 0, 0

    while step < int(total_steps):
        episode += 1
        done = False
        observation = env.reset()

        #  inner while loop runs through single episode
        while not done:
            step += 1

            action = agent.act(observation)

            next_observation, reward, done, info = env.step(action)

            agent.remember(observation, action, reward,
                           next_observation, done)
            runner.record_step(reward)

            observation = next_observation

            train_info = agent.learn()

        runner.record_episode(env_info=info)

        save_env_info(
            env,
            info,
            len(runner.episode_rewards),
            paths['env_histories']
        )

    return agent, env, runner


def test_experiment(
        sess,
        agent,
        env,
        runner,
        paths,
        total_steps,
        fill_memory=True
):
    """ experiment of multiple episodes with no learning or exploration """

    logger.info('starting test experiment of {} steps'.format(total_steps))

    #  outer while loop runs through multiple episodes
    step, episode = 0, 0

    while step < int(total_steps):
        episode += 1
        done = False
        observation = env.reset()

        #  inner while loop runs through a single episode
        while not done:
            step += 1

            action = agent.act(observation, explore=0)

            next_observation, reward, done, info = env.step(action)

            if fill_memory:
                agent.remember(observation, action, reward,
                               next_observation, done)

            runner.record_step(reward)

            observation = next_observation

        runner.record_episode(env_info=info)

        save_env_info(
            env,
            info,
            len(runner.episode_rewards),
            paths['env_histories']
        )

    return agent, env, runner


def single_run():
    """
    Perform a single run from an experiment

    To run the example experiment
        python experiment.py example dqn

    Protection for variable types is made in the env or the agent inits
    """
    pass


if __name__ == '__main__':
    # run_experiment()
    args = make_config_parser()

    #  cwd to avoid looking where package is installed
    #  could be a better fix for this TODO 
    paths = make_paths(
        os.getcwd(),
        args.expt_name,
        args.run_name
    )

    logger = make_logger(paths, name=__name__)

    run_config = parse_ini(paths['run_configs'], args.run_name)
    env_config = parse_ini(paths['expt_config'], 'env')
    expt_config = parse_ini(paths['expt_config'], 'expt')

    total_steps = int(run_config['total_steps'])

    train_steps = int(expt_config['train_steps'])
    test_steps = int(expt_config['test_steps'])

    #  could pop this in the setup_expt
    seed = run_config.pop('seed', None)

    #  could hve a way to copy run config args into the env
    #  sometimes we want to change more than the agent/seed - ie episode sampling
    #  is differnt for learning and non-learning agents - TODO as needed

    tf.reset_default_graph()

    with tf.Session() as sess:

        agent, env = setup_experiment(
            sess,
            run_config,
            env_config,
            paths,
            seed=seed
        )

        def pre_train(agent, pre_train_steps):
            assert len(agent.memory) > 10

            logging('pretraining agent for {} steps'.format(
                pre_train_steps))

            pt_step = 0
            while pt_step < pre_train_steps:
                agent.learn()

            return agent

        pre_train_steps = run_config.pop('pre_train_steps', None)

        if pre_train_steps:
            agent = pre_train(agent, pre_train_steps)

        steps = 0
        runner = Runner(sess, paths)

        while steps < total_steps:

            agent, env, runner = training_experiment(sess, agent, env, runner, paths, train_steps)

            agent, env, runner = test_experiment(sess, agent, env, runner, paths, test_steps)

            steps += train_steps + test_steps

        agent.memory.save(paths['memory'])

        process_experiment(args.expt_name, args.run_name)
