from energypy.experiment import ExperimentConfig, run_experiments


def test_run_experiments() -> None:
    configs = [
        ExperimentConfig(
            env_tr={"id": "energypy/battery"},
            env_te=None,
            agent={"id": "PPO", "policy": "MlpPolicy"},
            n_learning_steps=10,
        ),
        ExperimentConfig(
            n_learning_steps=10,
        ),
        ExperimentConfig(n_learning_steps=10),
    ]
    results = run_experiments(configs, log_dir=None)
    assert len(results) == len(configs)
