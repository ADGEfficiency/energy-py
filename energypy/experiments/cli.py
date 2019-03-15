import click
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import energypy
from energypy import read_log

from energypy import setup_expt, setup_run, perform_run


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


