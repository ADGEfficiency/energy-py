import click
import tensorflow as tf

import energypy as ep


@click.command()
@click.argument('expt', type=click.File('rb'))
@click.argument('run', nargs=1)
def cli(expt, run):

    with tf.Session() as sess:
        cfg = ep.setup_expt(expt)

        run_cfg, agent, env, runner = ep.setup_run(cfg, run, sess)

        ep.perform_run(runner, run_cfg, agent, env)

        ep.analyze_run(run_cfg)

    ep.analyze_experiment(cfg)
