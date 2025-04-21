"""Tools for running reinforcement learning experiments with energypy."""

from typing import Any, Dict
import numpy as np
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

import pydantic


class ExperimentResult(pydantic.BaseModel):
    mean_reward_tr: float
    mean_reward_te: float
    std_reward_tr: float
    std_reward_te: float


def run_episode(env, model) -> dict:
    """Interact with the environment using the trained model and display results."""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0

    infos = []
    while not done:
        # Act: Get the action from the model
        # TODO - should deterministic only be when in test mode?
        action, _states = model.predict(obs, deterministic=True)

        # Step: Execute the action in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated

        # Observe reward
        total_reward += reward

        # TODO - debug logging
        # print(f"Episode {episode + 1}, Step {step_counter + 1}")
        # print(f"  Observation: {obs}")
        # print(f"  Action: {action}")
        # print(f"  Reward: {reward}")
        # print(f"  Done: {done}")
        # print("---")

        # Update observation
        obs = next_obs
        step_counter += 1

    return {"total_reward": total_reward, "n_steps": step_counter, "infos": infos}


class ExperimentConfig(pydantic.BaseModel):
    env: Env[Any, Any]
    eval_env: Env[Any, Any]
    model: BaseAlgorithm
    name: str = "battery"
    num_episodes: int = 5

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


def run_experiment(
    cfg: ExperimentConfig | None = None, **kwargs: int
) -> ExperimentResult:
    # TODO - test all these
    # TODO - tests using the config
    """
    run_experiment(Config(options=1))
    run_experiment(cfg=Config(options=1))
    run_experiment(options=1)
    """
    if cfg is None:
        cfg = ExperimentConfig(**kwargs)

    assert isinstance(cfg, ExperimentConfig)

    cfg.model.learn(total_timesteps=50000)

    model_path = f"models/{cfg.name}"
    cfg.model.save(model_path)
    # TODO
    # model = PPO.load(model_path)

    results_tr: list[dict] = []
    for episode in range(cfg.num_episodes):
        results = run_episode(cfg.env, cfg.model).dict()
        print(
            f"Episode {episode + 1} completed with total reward: {results['total_reward']}, steps: {results['n_steps']}"
        )
        print("=" * 50)
        results_tr.append(results)

    mean_reward_te, std_reward_te = evaluate_policy(
        cfg.model, cfg.eval_env, n_eval_episodes=10, deterministic=True
    )

    result = ExperimentResult(
        mean_reward_tr=float(np.mean([r["total_reward"] for r in results_tr])),
        std_reward_tr=float(np.std([r["total_reward"] for r in results_tr])),
        mean_reward_te=float(mean_reward_te),
        std_reward_te=float(std_reward_te),
    )
    print(result)
    return result
