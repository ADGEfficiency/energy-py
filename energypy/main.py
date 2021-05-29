from collections import defaultdict
import random
from random import choice
from time import sleep
import time

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from energypy import alpha, checkpoint, json_util, init
from energypy import alpha, memory, policy, qfunc, random_policy, target, utils
from energypy.sampling import sample_random, sample_test, sample_train
from energypy.train import train


def main(
    hyp,
    paths,
    counters,
    env,
    buffer,
    nets,
    writers,
    optimizers,
    transition_logger,
    rewards
):
    if 'seed' not in hyp.keys():
        hyp['seed'] = choice(range(int(1e4)))

    utils.set_seeds(hyp['seed'])

    json_util.save(hyp, paths['run'] / 'hyperparameters.json')

    if not buffer.full:
        sample_random(
            env,
            buffer,
            hyp,
            writers,
            counters,
            rewards,
            transition_logger,
        )
        memory.save(buffer, paths['run'] / 'random.pkl')
        memory.save(buffer, paths['experiment'] / 'random.pkl')

    rewards = defaultdict(list)
    for _ in range(int(hyp['n-episodes'])):
        if counters['train-episodes'] % hyp['test-every'] == 0:
            test_rewards = sample_test(
                env,
                buffer,
                nets['actor'],
                hyp,
                writers,
                counters,
                rewards,
                transition_logger
            )

            checkpoint.save(
                hyp,
                nets,
                optimizers,
                buffer,
                episode=counters['test-episodes'],
                rewards=rewards,
                counters=counters,
                paths=paths
            )

        train_rewards = sample_train(
            env,
            buffer,
            nets['actor'],
            hyp,
            writers,
            counters,
            rewards,
            transition_logger
        )

        train_steps = len(train_rewards) * hyp.get('episode_length', 48)

        print(f'training \n step {counters["train-steps"]:6.0f}, {train_steps} steps')
        for _ in tqdm(range(train_steps)):
            train(
                buffer.sample(hyp['batch-size']),
                nets['actor'],
                [nets['online-1'], nets['online-2']],
                [nets['target-1'], nets['target-2']],
                nets['alpha'],
                writers['train'],
                optimizers,
                counters,
                hyp
            )
        utils.print_counters(counters)

    if counters['train-episodes'] % hyp['test-every'] == 0:
        test_rewards = sample_test(
            env,
            buffer,
            nets['actor'],
            hyp,
            writers,
            counters,
            rewards,
            transition_logger
        )

        checkpoint.save(
            hyp,
            nets,
            optimizers,
            buffer,
            episode=counters['test-episodes'],
            rewards=rewards,
            counters=counters,
            paths=paths
        )


@click.command()
@click.argument("experiment-json", nargs=1)
@click.option("-n", "--run-name", default=None)
@click.option("-b", "--buffer", nargs=1, default="new")
@click.option("-s", "--seed", nargs=1, default=None)
@click.option("-c", "--checkpoint_path", nargs=1, default=None)
def cli(experiment_json, run_name, buffer, seed, checkpoint_path):

    print('cli')
    print('------')
    print(experiment_json, run_name, buffer)
    print('')

    hyp = json_util.load(experiment_json)
    hyp['buffer'] = buffer

    if run_name:
        hyp['run-name'] = run_name

    print('params')
    print('------')
    print(hyp)
    print('')
    sleep(2)

    if checkpoint_path:
        print(f'checkpointing from {checkpoint_path}')
        print('')
        main(**init.init_checkpoint(checkpoint_path))

    else:
        print(f'starting so fresh, so clean')
        print('')
        main(**init.init_fresh(hyp))


if __name__ == '__main__':
    cli()
