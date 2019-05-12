import json
from os import makedirs
from os.path import expanduser, join
from shutil import copyfile

import numpy as np
import yaml

from energypy import make_new_logger, make_env, make_agent
from energypy.common import dump_config
from energypy.experiments.utils import Runner


def setup_expt(expt, ftype='yaml'):
    """ creates experiment config dict """
    if ftype == 'yaml':
        cfg = yaml.load(expt)
    else:
        cfg = expt

    home = expanduser('~')
    expt_dir = '{}/energy-py-results/{}'.format(home, cfg['expt']['name'])
    makedirs(expt_dir, exist_ok=True)

    expt_logger = make_new_logger('expt', expt_dir)

    cfg['expt']['expt_dir'] = expt_dir

    dump_config(cfg, expt_logger)

    return cfg


def make_run_config(expt_cfg, run):
    run_cfg = expt_cfg[run]

    run_dir = join(expt_cfg['expt']['expt_dir'], run)
    makedirs(run_dir, exist_ok=True)
    run_cfg['run_dir'] = run_dir

    ep_dir = join(run_dir, 'episodes')
    makedirs(ep_dir, exist_ok=True)
    run_cfg['ep_dir'] = ep_dir

    tensorboard_dir = join(expt_cfg['expt']['expt_dir'], 'tensorboard', run)
    makedirs(tensorboard_dir, exist_ok=True)
    run_cfg['tensorboard_dir'] = tensorboard_dir

    return run_cfg


def setup_run(cfg, run, sess):
    run_cfg = make_run_config(cfg, run)

    run_logger = make_new_logger('run_setup', run_cfg['run_dir'])
    runner = Runner(sess, run_cfg)
    dump_config(run_cfg, run_logger)

    env_config = run_cfg['env']
    env = make_env(**env_config)

    if hasattr(env.observation_space, 'info') and hasattr(env.state_space, 'info'):
        run_logger.debug(json.dumps({'state_info': env.state_space.info}))
        run_logger.debug(json.dumps({'observation_info': env.observation_space.info}))

    agent_config = run_cfg['agent']
    agent_config['env'] = env
    agent_config['sess'] = sess
    agent_config['tensorboard_dir'] = run_cfg['tensorboard_dir']

    agent = make_agent(**agent_config)

    return run_cfg, agent, env, runner


def perform_run(runner, run_cfg, agent, env):
    total_steps = run_cfg['total_steps']
    #  pretraining would go here

    step, episode = 0, 0
    while step < int(total_steps):
        episode += 1
        env.episode_logger = make_new_logger(
            'ep_{}'.format(episode),
            run_cfg['ep_dir']
        )

        info = perform_episode(agent, env)
        runner.record_episode(info['rewards'])
        step += info['steps']


def perform_episode(agent, env):
    done = False
    step = 0
    rewards = []
    observation = env.reset()
    while not done:
        step += 1
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, next_observation, done)
        rewards.append(reward)
        observation = next_observation

        #  only learn once we have 5000 samples
        if len(agent.memory) > 5000:
            agent.learn()

    info['steps'] = step

    return info
