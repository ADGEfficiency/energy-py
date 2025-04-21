"""Tools for running reinforcement learning experiments with energypy."""

from typing import Any

import gymnasium as gym
import numpy as np
import pydantic
import stable_baselines3
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

import energypy


def _get_default_battery():
    return energypy.Battery(electricity_prices=np.random.uniform(-100, 100, 48 * 10))


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
        if isinstance(v["env_tr"], dict):
            env = gym.make(**v["env_tr"])
            v["env_tr"] = env

        if v["env_te"] is None:
            v["env_te"] = v["env_tr"]

        # TODO - more init of env_te if not None

        if isinstance(v["agent"], dict):
            Agent = getattr(stable_baselines3, v["agent"]["id"])
            del v["agent"]["id"]
            v["agent"] = Agent(**v["agent"], env=v["env_tr"])

        return v


class ExperimentResult(pydantic.BaseModel):
    mean_reward_tr: float
    mean_reward_te: float
    std_reward_tr: float
    std_reward_te: float


def run_experiment(
    cfg: ExperimentConfig | None = None, **kwargs: str | int
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

    cfg.agent.learn(total_timesteps=cfg.n_learning_steps)

    model_path = f"models/{cfg.name}"
    cfg.agent.save(model_path)
    # TODO
    # agent = PPO.load(model_path)

    mean_reward_tr, std_reward_tr = evaluate_policy(
        cfg.agent, cfg.env_tr, n_eval_episodes=cfg.n_eval_episodes, deterministic=True
    )

    mean_reward_te, std_reward_te = evaluate_policy(
        cfg.agent, cfg.env_te, n_eval_episodes=cfg.n_eval_episodes, deterministic=True
    )

    result = ExperimentResult(
        mean_reward_tr=float(mean_reward_tr),
        std_reward_tr=float(std_reward_tr),
        mean_reward_te=float(mean_reward_te),
        std_reward_te=float(std_reward_te),
    )
    print(result)
    return result


def run_episode(env, agent) -> dict:
    """Interact with the environment using the trained agent and display results."""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0

    infos = []
    while not done:
        # Act: Get the action from the agent
        # TODO - should deterministic only be when in test mode?
        action, _states = agent.predict(obs, deterministic=True)

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

    # TODO - EpisodeResult
    return {"total_reward": total_reward, "n_steps": step_counter, "infos": infos}
