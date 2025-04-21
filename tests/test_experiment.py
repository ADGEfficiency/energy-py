# TODO - test all these
# TODO - tests using the config
"""
run_experiment(Config(options=1))
run_experiment(cfg=Config(options=1))
run_experiment(options=1)
"""

import energypy
from energypy.experiment import ExperimentConfig, run_experiments


def test_experiment_from_json() -> None:
    cfg = ExperimentConfig(
        env_tr={"id": "energypy/battery"},
        env_te=None,
        agent={"id": "PPO", "policy": "MlpPolicy"},
    )
    energypy.run_experiment(cfg)


def test_experiment() -> None:
    cfg = ExperimentConfig()
    energypy.run_experiment(cfg)


def test_multiple_experiments() -> None:
    configs = [
        ExperimentConfig(n_learning_steps=10),
        ExperimentConfig(n_learning_steps=20),
    ]
    results = run_experiments(configs, log_dir="./data/tensorboard/test_experiments")
    assert len(results) == 2
