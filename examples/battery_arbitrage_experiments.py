import gymnasium as gym
import numpy as np
import polars as pl
from stable_baselines3 import PPO

import energypy
from energypy.experiment import ExperimentConfig

env_id = "energypy/battery"
gym.register(
    id=env_id,
    entry_point="energypy:Battery",
)

# TODO - download data if not already there
data = pl.read_parquet("data/final.parquet")
data = data.select(
    "datetime",
    pl.col("DollarsPerMegawattHour").alias("price"),
    pl.col("PointOfConnection").alias("point_of_connection"),
)

prices = data["price"]

features = prices.clone().to_frame()
features = features.with_columns(
    [pl.col("price").shift(n).alias(f"lag-{n}") for n in range(48)]
)
features = features.drop_nulls()

split_idx = int(data.shape[0] // 2)
prices_tr = prices.slice(0, split_idx)
prices_te = prices.slice(split_idx, data.shape[0])

features_tr = features.slice(0, split_idx)
features_te = features.slice(split_idx, data.shape[0])

import uuid

expt_guid = uuid.uuid4()

configs = []
for noise in [0, 1, 10, 100, 1000]:
    run_guid = uuid.uuid4()
    env_tr = gym.wrappers.NormalizeReward(
        gym.make(env_id, electricity_prices=prices_tr, features=features)
    )
    env_te = gym.wrappers.NormalizeReward(
        gym.make(
            env_id,
            electricity_prices=prices_te,
            features=prices_te * np.random.normal(0, noise, size=prices_te.shape[0]),
        )
    )

    config = ExperimentConfig(
        env_tr=env_tr,
        env_te=env_te,
        agent=PPO(
            policy="MlpPolicy",
            env=env_tr,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=f"./data/tensorboard/battery_arbitrage_experiments/{expt_guid}/run/{run_guid}",
        ),
        name=f"battery_noise_{noise}",
        n_learning_steps=5000,  # Short training for demonstration
        n_eval_episodes=25,
    )
    configs.append(config)

# Run all experiments and log to tensorboard
results = energypy.run_experiments(
    configs, log_dir=f"./data/tensorboard/battery_arbitrage_experiments/{expt_guid}"
)

# Print the best performing configuration on test data
best_idx = np.argmax([r.checkpoints[-1].mean_reward_te for r in results])
best_config = configs[best_idx]
best_result = results[best_idx].checkpoints[-1]

print(f"Best configuration: {best_config.name}")
print(f"Learning rate: {best_config.agent.learning_rate}")
print(f"Gamma: {best_config.agent.gamma}")
print(
    f"Test reward: {best_result.mean_reward_te:.2f} ± {best_result.std_reward_te:.2f}"
)
print(
    f"Train reward: {best_result.mean_reward_tr:.2f} ± {best_result.std_reward_tr:.2f}"
)
