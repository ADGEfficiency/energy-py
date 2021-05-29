from energypy.main import main
from energypy.init import init_fresh

from pathlib import Path
from shutil import rmtree

hyp = {
    'initial-log-alpha': 0.0,
    'gamma': 0.99,
    'rho': 0.995,
    'buffer-size': int(1e2),
    'reward-scale': 5,
    'lr': 3e-4,
    'batch-size': 1,
    'n-episodes': 1,
    'test-every': 1,
    'n-tests': 1,
    'size-scale': 1,
    'env': {
      'name': 'pendulum',
      'env_name': 'pendulum'
    },
    'run-name': 'test-system',
    'buffer': 'new'
}

batt_hyp = hyp.copy()
batt_hyp['env'] = {
    'name': 'battery',
    'dataset': {'name': 'random-dataset'},
    'n_batteries': 2
}


def test_system():
    main(**init_fresh(hyp))
    run_path = './experiments/pendulum/test-system'
    print(f'deleting {run_path}\n')
    rmtree(str(run_path))
    return run_path


def test_system_battery():
    main(**init_fresh(batt_hyp))
    run_path = './experiments/battery/test-system'
    print(f'deleting {run_path}\n')
    rmtree(str(run_path))
    return run_path
