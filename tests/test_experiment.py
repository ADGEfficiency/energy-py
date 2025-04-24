from energypy.experiment import ExperimentConfig, run_experiment, run_experiments


def test_run_experiments() -> None:
    """Test running multiple experiments with different configurations."""
    configs = [
        ExperimentConfig(
            env_tr={"id": "energypy/battery"},
            env_te=None,
            agent={"id": "PPO", "policy": "MlpPolicy"},
            n_learning_steps=10,
        ),
        ExperimentConfig(
            env_tr={"id": "energypy/battery"},
            env_te={"id": "energypy/battery"},
            n_learning_steps=10,
        ),
        ExperimentConfig(n_learning_steps=10),
    ]
    results = run_experiments(configs, log_dir=None)
    assert len(results) == len(configs)


def test_run_experiment_from_kwargs() -> None:
    """Test running a single experiment using keyword arguments instead of ExperimentConfig."""
    _ = run_experiment(
        env_tr={"id": "energypy/battery"},
        env_te=None,
        agent={"id": "PPO", "policy": "MlpPolicy"},
        n_learning_steps=10,
    )
