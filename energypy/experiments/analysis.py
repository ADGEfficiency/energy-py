import matplotlib.pyplot as plt
import json
import pandas as pd
from os.path import join

from energypy.experiments.utils import read_log
from energypy.experiments.blocks import make_run_config

def read_results(cfg):
    return {
        'setup': read_log(join(cfg['run_dir'], 'run_setup.log')),
        'rewards': read_log(join(cfg['run_dir'], 'results.log')),
        'episode_1': read_log(join(cfg['run_dir'], 'episodes', 'ep_1.log'))
    }


def plot_results_log(run_cfg, ax):
    episode_rewards = read_log(join(run_cfg['run_dir'], 'results.log'))
    episode_rewards = pd.DataFrame(episode_rewards)
    episode_rewards.plot(ax=ax, x='episode', y='rew_100', label='avg_100_episodes')
    episode_rewards.plot(ax=ax, x='episode', y='reward', label='episode_reward')


def analyze_run(run_cfg):

    f, ax = plt.subplots()

    plot_results_log(run_cfg, ax)

    f.savefig(join(run_cfg['run_dir'], 'episode_rewards.png'))


def analyze_experiment(expt_cfg):

    runs = [run for run in expt_cfg.keys() if 'run' in run]

    fig, axes = plt.subplots(nrows=len(runs))

    for ax, run in zip(axes, runs):
        run_cfg = make_run_config(expt_cfg, run)

        try:
            plot_results_log(run_cfg, ax)

        except FileNotFoundError:
            pass

    fig.savefig(join(expt_cfg['expt']['expt_dir'], 'all_runs.png'))


