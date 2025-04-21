import energypy
from energypy.experiment import ExperimentConfig


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
