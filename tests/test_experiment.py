from energypy.experiment import ExperimentConfig, run_experiments


def test_run_experiments() -> None:
    configs = [
        ExperimentConfig(
            env_tr={"id": "energypy/battery"},
            env_te=None,
            agent={"id": "PPO", "policy": "MlpPolicy"},
        ),
        ExperimentConfig(),
        ExperimentConfig(n_learning_steps=20),
    ]
    results = run_experiments(configs, log_dir="./data/tensorboard/test_experiments")
    assert len(results) == 2
