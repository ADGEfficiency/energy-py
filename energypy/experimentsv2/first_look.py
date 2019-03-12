import logging.config
import json
from shutil import copyfile
import os
from os.path import expanduser, join

import click
import tensorflow as tf
import yaml

import energypy


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def dump_config(cfg, logger):
    for k, v in cfg.items():
        logger.info(json.dumps({k: v}))


def make_new_logger(log_dir, name):
    logger = logging.getLogger(name)

    fmt = '%(asctime)s [%(levelname)s]%(name)s: %(message)s'

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,

            'formatters': {
                'standard': {'format': fmt, 'datefmt': '%Y-%m-%dT%H:M:S'},
                'file': {'format': '%(message)s'}
            },

            'handlers': {
                'console': {
                    'level': 'INFO',
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard'},

                'file': {
                     'class': 'logging.FileHandler',
                     'level': 'DEBUG',
                     'filename': join(log_dir, '{}.log'.format(name)),
                     'mode': 'w',
                     'formatter': 'file'}
            },

            'loggers': {
                '': {
                  'handlers': ['console', 'file'],
                  'level': 'DEBUG',
                  'propagate': True}
            }
        }
    )

    return logger


@click.command()
@click.argument('expt', type=click.File('rb'))
@click.argument('run', nargs=1)
def cli(expt, run):

    with tf.Session() as sess:
        cfg = setup_expt(expt)

        run_cfg, agent, env = setup_run(cfg, run, sess)

        perform_run(run_cfg, agent, env)


def setup_expt(expt):
    cfg = yaml.load(expt)

    home = expanduser('~')
    expt_dir = '{}/energypy-results/{}'.format(home, cfg['expt']['name'])
    ensure_dir(expt_dir)

    expt_logger = make_new_logger(expt_dir, 'expt')

    cfg['expt']['expt_dir'] = expt_dir

    dump_config(cfg, expt_logger)

    return cfg


def setup_run(cfg, run, sess):
    run_cfg = cfg[run]

    run_dir = os.path.join(cfg['expt']['expt_dir'], run)
    ensure_dir(run_dir)
    run_cfg['run_dir'] = run_dir

    ep_dir = os.path.join(run_dir, 'episodes')
    ensure_dir(ep_dir)
    run_cfg['ep_dir'] = ep_dir

    tensorboard_dir = os.path.join(cfg['expt']['expt_dir'], 'tensorboard', run)
    ensure_dir(tensorboard_dir)

    run_logger = make_new_logger(run_dir, 'run_info')

    dump_config(run_cfg, run_logger)

    env_config = run_cfg['env']
    env = energypy.make_env(**env_config)

    if hasattr(env.observation_space, 'info') and hasattr(env.state_space, 'info'):
        run_logger.debug(json.dumps({'state_info': env.state_space.info}))
        run_logger.debug(json.dumps({'observation_info': env.observation_space.info}))

    agent_config = run_cfg['agent']
    agent_config['env'] = env
    agent_config['sess'] = sess
    agent_config['tensorboard_dir'] = tensorboard_dir

    agent = energypy.make_agent(**agent_config)

    return run_cfg, agent, env


def perform_run(run_cfg, agent, env):

    total_steps = run_cfg['total_steps']

    step, episode = 0, 0
    while step < int(total_steps):
        episode += 1
        env.episode_logger = make_new_logger(
            run_cfg['ep_dir'], 'ep_{}'.format(episode)
        )

        episode_steps = perform_episode(agent, env)
        step += episode_steps


def perform_episode(agent, env):
    done = False
    observation = env.reset()
    step = 0

    while not done:
        step += 1

        action = agent.act(observation)

        next_observation, reward, done, info = env.step(action)

        agent.remember(
            observation, action, reward, next_observation, done
        )

        observation = next_observation

        #  only learn once memory is full
        if len(agent.memory) > min(agent.memory.size, 10000):
            agent.learn()

    return step
