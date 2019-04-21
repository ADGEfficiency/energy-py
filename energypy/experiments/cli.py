import click
import tensorflow as tf

from energypy.experiments.blocks import setup_expt, setup_run, perform_run, make_run_config
from energypy.experiments.analysis import analyze_run, analyze_experiment


@click.command()
@click.argument('expt', type=click.File('rb'))
@click.argument('run', nargs=1)
def cli(expt, run):

    with tf.Session() as sess:
        cfg = setup_expt(expt)

        run_cfg, agent, env, runner = setup_run(cfg, run, sess)

        perform_run(runner, run_cfg, agent, env)

        analyze_run(run_cfg)

    analyze_experiment(cfg)
