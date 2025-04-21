import gymnasium as gym
import polars as pl
import numpy as np
from stable_baselines3 import PPO

import energypy

env_id = "energypy/battery"
gym.register(
    id=env_id,
    entry_point="energypy:Battery",
)

# TODO - download data if not already there
data = pl.read_parquet("data/final.parquet")
prices = data["DollarsPerMegawattHour"]
prices = np.random.uniform(-1000, 1000, 2048 * 10)
env = gym.make(env_id, electricity_prices=prices)
env = gym.wrappers.NormalizeReward(env)

result = energypy.run_experiment(
    env=env,
    eval_env=env,
    model=PPO(
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
    ),
    name="cartpole",
).dict()
assert isinstance(result["mean_reward_te"], float) and result["mean_reward_te"] > 4.0
