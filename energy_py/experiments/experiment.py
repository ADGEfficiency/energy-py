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

    #  TODO copy the dataset into the run folder as well

    return agent, env


def experiment(
        sess,
        agent,
        env,
        runner,
        paths,
        total_steps,
        mode='training',
):
    """ experiment of multiple episodes with learning """

    logger.info('starting {} experiment of {} steps'.format(mode, total_steps))

    if mode == 'training':
        explore = 1.0
    elif mode == 'testing':
        explore = 0.0
    else:
        raise ValueError('mode of {} not supported'.format(mode))

    #  outer while loop runs through multiple episodes
    step, episode = 0, 0

    while step < int(total_steps):
        episode += 1
        done = False
        observation = env.reset()

        #  inner while loop runs through single episode
        while not done:
            step += 1

            action = agent.act(observation, explore=explore)

            next_observation, reward, done, info = env.step(action)

            agent.remember(observation, action, reward,
                           next_observation, done)

            runner.record_step(reward)

            observation = next_observation

            if mode == 'training':
                train_info = agent.learn()

        runner.record_episode(env_info=info)

        save_env_info(
            env,
            info,
            len(runner.episode_rewards),
            paths['env_histories']
        )

    return agent, env, runner


def pre_train(agent, pre_train_steps):
    assert len(agent.memory) > 10

    logging('pretraining agent for {} steps'.format(
        pre_train_steps))

    pt_step = 0
    while pt_step < pre_train_steps:
        agent.learn()

    return agent


def single_run():
    """
    Perform a single run from an experiment

    To run the example experiment
        python experiment.py example dqn

    Protection for variable types is made in the env or the agent inits
    """
    pass


if __name__ == '__main__':
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

        pre_train_steps = run_config.pop('pre_train_steps', None)

        if pre_train_steps:
            agent = pre_train(agent, pre_train_steps)

        steps = 0
        runner = Runner(sess, paths)

        while steps < total_steps:
            agent, env, runner = experiment(
                sess, agent, env, runner, paths, train_steps
            )

            if train_steps > 0:
                agent, env, runner = experiment(
                    sess, agent, env, runner, paths, test_steps, mode='testing'
                )

            steps += train_steps + test_steps

        agent.memory.save(paths['memory'])

        process_experiment(args.expt_name, args.run_name)
