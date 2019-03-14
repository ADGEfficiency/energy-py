import logging
import json
from shutil import copyfile
from os.path import expanduser, join

import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

import energypy
from .logging import make_new_logger
from .utils import load_dataset, ensure_dir, dump_config, read_log


@click.command()
@click.argument('expt', type=click.File('rb'))
@click.argument('run', nargs=1)
def cli(expt, run):

    with tf.Session() as sess:
        cfg = setup_expt(expt)

        run_cfg, agent, env, runner = setup_run(cfg, run, sess)

        perform_run(runner, run_cfg, agent, env)

        def analyze_run(run_cfg):

            episode_rewards = read_log(join(run_cfg['run_dir'], 'results.log'))
            episode_rewards = pd.DataFrame(episode_rewards)

            f, ax = plt.subplots()
            episode_rewards.plot(ax=ax, x='episode', y='rew_100', label='avg_100_episodes')
            episode_rewards.plot(ax=ax, x='episode', y='reward', label='episode_reward')

            f.savefig(join(run_cfg['run_dir'], 'episode_rewards.png'))
            return ax

        analyze_run(run_cfg)

        def analyze_expt(expt_cfg):

            import pdb; pdb.set_trace()

            runs = expt_cfg.keys()

            fig, axes = plt.subplots(nrows=len(runs))

            for idx, run in enumerate(runs):
                #  ignore 'expt'
                if 'run' in run:
                    run_cfg = make_run_config(expt_cfg, run)
                    try:
                        ax = analyze_run(run_cfg)
                        axes [idx] = ax
                    except FileNotFoundError:
                        pass

    analyze_expt(cfg)


def setup_expt(expt):
    cfg = yaml.load(expt)

    home = expanduser('~')
    expt_dir = '{}/energypy-results/{}'.format(home, cfg['expt']['name'])
    ensure_dir(expt_dir)

    expt_logger = make_new_logger('expt', expt_dir)

    cfg['expt']['expt_dir'] = expt_dir

    dump_config(cfg, expt_logger)

    return cfg

def make_run_config(expt_cfg, run):
    run_cfg = expt_cfg[run]

    run_dir = join(expt_cfg['expt']['expt_dir'], run)
    ensure_dir(run_dir)
    run_cfg['run_dir'] = run_dir

    ep_dir = join(run_dir, 'episodes')
    ensure_dir(ep_dir)
    run_cfg['ep_dir'] = ep_dir

    tensorboard_dir = join(expt_cfg['expt']['expt_dir'], 'tensorboard', run)
    ensure_dir(tensorboard_dir)
    run_cfg['tensorboard_dir'] = tensorboard_dir

    return run_cfg


def setup_run(cfg, run, sess):

    run_cfg = make_run_config(cfg, run)

    run_logger = make_new_logger('run_setup', run_cfg['run_dir'])
    runner = Runner(sess, run_cfg)
    dump_config(run_cfg, run_logger)

    env_config = run_cfg['env']
    env = energypy.make_env(**env_config)

    if hasattr(env.observation_space, 'info') and hasattr(env.state_space, 'info'):
        run_logger.debug(json.dumps({'state_info': env.state_space.info}))
        run_logger.debug(json.dumps({'observation_info': env.observation_space.info}))

    agent_config = run_cfg['agent']
    agent_config['env'] = env
    agent_config['sess'] = sess
    agent_config['tensorboard_dir'] = run_cfg['tensorboard_dir']

    agent = energypy.make_agent(**agent_config)

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

        episode_steps, episode_rewards = perform_episode(agent, env)
        runner.record_episode(episode_rewards)
        step += episode_steps


def perform_episode(agent, env):
    done = False
    observation = env.reset()
    step = 0
    rewards = []

    while not done:
        step += 1

        action = agent.act(observation)

        next_observation, reward, done, info = env.step(action)

        agent.remember(observation, action, reward, next_observation, done)
        rewards.append(reward)

        observation = next_observation

        #  only learn once memory is full
        if len(agent.memory) > min(agent.memory.size, 10000):
            agent.learn()

    return step, rewards

class Runner():
    """ logs episode reward stats """

    def __init__(self, sess, run_cfg):

        self.writer = tf.summary.FileWriter(
            run_cfg['tensorboard_dir'], sess.graph
        )

        self.logger = make_new_logger('results', run_cfg['run_dir'])

        self.reset()

    def reset(self):
        self.history = []
        self.step = 0

    def record_episode(self, episode_rewards):

        total_episode_reward = np.sum(episode_rewards)
        self.history.append(total_episode_reward)

        summaries = {
            'total_episode_reward': total_episode_reward,
            'avg_rew_100': np.mean(self.history[-100:]),
            'min_rew_100': np.min(self.history[-100:]),
            'max_rew_100': np.max(self.history[-100:]),
            'avg_rew': np.mean(self.history),
            'min_rew': np.min(self.history),
            'max_rew': np.max(self.history)
        }

        for tag, value in summaries.items():
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=float(value))])
            self.writer.add_summary(summary, self.step)

        self.writer.flush()

        log = {
            'episode': len(self.history), 
            'reward': total_episode_reward, 
            'rew_100': summaries['avg_rew_100']
        }

        self.logger.info(json.dumps(log))
