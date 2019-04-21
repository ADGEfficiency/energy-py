import csv
import os
from os.path import join
import shutil

import numpy as np
import pytest
import tensorflow as tf

from energypy.experiments.analysis import read_results
from energypy.experiments.blocks import *
from energypy.experiments.utils import read_log

#  CLI test TODO
# os.system('energypy-experiment examples/example_config.yaml cartpole')

#  fixture - make test data folder, 
#  kill test data folder AND results folder off home

test_dir = './tmp-test-data'


@pytest.fixture()
def setup_and_teardown():
    shutil.rmtree('/Users/adam/energy-py-results/test', ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)

    yield

    shutil.rmtree('/Users/adam/energy-py-results/test', ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)


def test_battery_random_expt(setup_and_teardown):

    cfg = {
        'expt': {
            'name': 'test',
            'total_steps': 8
        },
        'test-run': {
            'total_steps': 16,
            'env': {'env_id': 'battery', 'dataset': test_dir, 'sample_strat': 'full'},
            'agent': {'agent_id': 'random'}
        }
    }

    dataset = [
        ['date', 'price [$/MWh]'],
        ['first', '90'],
        ['second', '80'],
        ['third', '70'],
        ['fourth', '80'],
        ['fifth', '90']
    ]

    with open(join(test_dir, 'state.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(dataset)

    with tf.Session() as sess:
        cfg = setup_expt(cfg, ftype='dict')
        run_cfg, agent, env, runner = setup_run(cfg, 'test-run', sess)

        perform_run(runner, run_cfg, agent, env)

    output = read_results(run_cfg)

    ep1 = output['episode_1']
    all_prices = [float(tup[1]) for tup in dataset[1:]]
    prices = [float(transition['electricity_price']) for transition in ep1]
    np.testing.assert_array_equal(all_prices, prices)


def test_cartpole_dqn(setup_and_teardown):

    cfg = {
        'expt': {
            'name': 'test',
        },
        'test-run': {
            'total_steps': 400,
            'env': {'env_id': 'cartpole-v0'},
            'agent': {'agent_id': 'random'}
        }
    }

    with tf.Session() as sess:
        cfg = setup_expt(cfg, ftype='dict')
        run_cfg, agent, env, runner = setup_run(cfg, 'test-run', sess)

        perform_run(runner, run_cfg, agent, env)

    output = read_results(run_cfg)
