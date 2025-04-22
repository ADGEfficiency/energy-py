"""Tools for running reinforcement learning experiments with energypy."""

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pydantic
import stable_baselines3
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

import energypy


def _get_default_battery():
    return energypy.Battery(electricity_prices=np.random.uniform(-100.0, 100, 48 * 10))


def _get_default_agent():
    env = _get_default_battery()
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./data/tensorboard",
    )


class ExperimentConfig(pydantic.BaseModel):
    env_tr: Env[Any, Any] = pydantic.Field(default_factory=_get_default_battery)
    env_te: Env[Any, Any] | None = None
    agent: BaseAlgorithm = pydantic.Field(default_factory=lambda: _get_default_agent())
    name: str = "battery"
    num_episodes: int = 10
    n_learning_steps: int = 50000
    n_eval_episodes: int = 10
    model_config: pydantic.ConfigDict = pydantic.ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    @pydantic.model_validator(mode="before")
    def validate_all_the_things(cls, v, values):
        if isinstance(v.get("env_tr"), dict):
            env = gym.make(**v["env_tr"])
            v["env_tr"] = env

        if (v.get("env_te") is None) and v.get("env_tr" is not None):
            v["env_te"] = v["env_tr"]

        # TODO - more init of env_te if not None

        if isinstance(v.get("agent"), dict):
            Agent = getattr(stable_baselines3, v["agent"]["id"])
            del v["agent"]["id"]
            v["agent"] = Agent(**v["agent"], env=v["env_tr"])

        return v

    @pydantic.model_validator(mode="after")
    def validate_all_the_things_again(cls, v, values):
        if v.env_te is None:
            v.env_te = v.env_tr
        return v


class Checkpoint(pydantic.BaseModel):
    learning_steps: int
    mean_reward_tr: float
    mean_reward_te: float
    std_reward_tr: float
    std_reward_te: float


class ExperimentResult(pydantic.BaseModel):
    checkpoints: list[Checkpoint] = []


def _evaluate_agent(
    agent: BaseAlgorithm,
    env_tr: Env[Any, Any],
    env_te: Env[Any, Any],
    n_eval_episodes: int,
    learning_steps: int = 0,
    deterministic: bool = True,
    callback: Any = None,
    writer: SummaryWriter | None = None,
) -> Checkpoint:
    """Evaluate an agent on training and test environments.

    Args:
        agent: The agent to evaluate
        env_tr: Training environment
        env_te: Test environment
        n_eval_episodes: Number of episodes to evaluate for
        learning_steps: Current learning steps completed
        deterministic: Whether to use deterministic actions
        callback: Optional callback for collecting additional information
        writer: Optional SummaryWriter for tensorboard logging

    Returns:
        Checkpoint with evaluation results
    """
    mean_reward_tr, std_reward_tr = evaluate_policy(
        agent,
        env_tr,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        callback=callback,
    )

    mean_reward_te, std_reward_te = evaluate_policy(
        agent,
        env_te,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )

    checkpoint = Checkpoint(
        learning_steps=learning_steps,
        mean_reward_tr=float(mean_reward_tr),
        std_reward_tr=float(std_reward_tr),
        mean_reward_te=float(mean_reward_te),
        std_reward_te=float(std_reward_te),
    )

    # Log to tensorboard if a writer is provided
    if writer is not None:
        writer.add_scalar(
            "Reward/train", checkpoint.mean_reward_tr, checkpoint.learning_steps
        )
        writer.add_scalar(
            "Reward/test", checkpoint.mean_reward_te, checkpoint.learning_steps
        )
        writer.add_scalar(
            "Reward_std/train", checkpoint.std_reward_tr, checkpoint.learning_steps
        )
        writer.add_scalar(
            "Reward_std/test", checkpoint.std_reward_te, checkpoint.learning_steps
        )

    return checkpoint


def run_experiment(
    cfg: ExperimentConfig | None = None,
    writer: SummaryWriter | None = None,
    experiment_index: int = 0,
    **kwargs: str | int,
) -> ExperimentResult:
    if cfg is None:
        cfg = ExperimentConfig(**kwargs)

    assert isinstance(cfg, ExperimentConfig)

    class Callback:
        import collections

        state = collections.defaultdict(list)

        def __call__(self, locals, globals):
            self.state["state_of_charge_mwh"].append(
                locals["info"]["state_of_charge_mwh"]
            )

    cb = Callback()

    # Evaluate agent before training
    checkpoint = _evaluate_agent(
        agent=cfg.agent,
        env_tr=cfg.env_tr,
        env_te=cfg.env_te,
        n_eval_episodes=cfg.n_eval_episodes,
        learning_steps=0,
        deterministic=True,
        callback=cb,
        writer=writer,
    )

    result = ExperimentResult(checkpoints=[checkpoint])

    # Train the agent
    cfg.agent.learn(total_timesteps=cfg.n_learning_steps)

    # Evaluate after training
    final_checkpoint = _evaluate_agent(
        agent=cfg.agent,
        env_tr=cfg.env_tr,
        env_te=cfg.env_te,
        n_eval_episodes=cfg.n_eval_episodes,
        learning_steps=cfg.n_learning_steps,
        deterministic=True,
        callback=cb,
        writer=writer,
    )

    # Add final checkpoint to results
    result.checkpoints.append(final_checkpoint)

    # Save the model
    model_path = f"models/{cfg.name}"
    cfg.agent.save(model_path)

    return result


def run_experiments(
    configs: Sequence[ExperimentConfig], log_dir: str = "./data/tensorboard/experiments"
) -> list[ExperimentResult]:
    """Run multiple experiments and log results to tensorboard.

    Args:
        configs: Sequence of experiment configurations
        log_dir: Directory for tensorboard logs

    Returns:
        List of experiment results
    """
    writer = SummaryWriter(log_dir=log_dir)
    results = []

    for i, cfg in enumerate(configs):
        result = run_experiment(cfg=cfg, writer=writer, experiment_index=i)
        results.append(result)

    writer.close()
    return results
