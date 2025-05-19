import collections
import uuid

import numpy as np
import polars as pl
from stable_baselines3 import PPO

import energypy
from energypy.dataset import load_electricity_prices

data = load_electricity_prices()

n_lags = 0
n_horizons = 12

data = data.with_columns(
    [pl.col("price").shift(n).alias(f"lag-{n}") for n in range(n_lags, n_lags + 1)]
)
data = data.with_columns(
    [
        pl.col("price").shift(-1 * n).alias(f"horizon-{n}")
        for n in range(1, n_horizons + 1)
    ]
)
data = data.drop_nulls()

prices = data["price"].to_numpy()
features = data.select(
    pl.selectors.starts_with("horizon-"), pl.selectors.starts_with("lag-")
).to_numpy()

te_tr_split_idx = int(data.shape[0] * 0.8)

prices_tr = prices[0:te_tr_split_idx]
features_tr = features[0:te_tr_split_idx]

prices_te = prices[te_tr_split_idx:]
features_te = features[te_tr_split_idx:]

print(f"prices_tr: {prices_tr.shape} features_tr: {features_tr.shape}")
print(f"prices_te: {prices_te.shape} features_te: {features_te.shape}")

expt_guid = uuid.uuid4()
configs = []
noise = [0.0, 0.1, 0.5, 0.75, 1, 5, 25, 100, 1000]
for noise_var in noise:
    run_guid = uuid.uuid4()
    env_tr = energypy.make_env(electricity_prices=prices_tr, features=features_tr)
    env_te = energypy.make_env(
        electricity_prices=prices_te,
        features=features_te * np.random.normal(0, noise_var, size=features_te.shape),
    )

    config = energypy.ExperimentConfig(
        env_tr=env_tr,
        env_te=env_te,
        agent=PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.000386,
            n_steps=1024,
            batch_size=64,
            n_epochs=2,
            gamma=0.92,
            gae_lambda=0.92,
            clip_range=0.21,
            policy_kwargs=dict(net_arch=[28] * 3),
            verbose=0,
            tensorboard_log=f"./data/tensorboard/battery_arbitrage_experiments/{expt_guid}/run/{run_guid}",
        ),
        name=f"battery_noise_{noise_var}",
        n_learning_steps=5000,
        n_eval_episodes=30,
    )
    configs.append(config)

results = energypy.run_experiments(
    configs, log_dir=f"./data/tensorboard/battery_arbitrage_experiments/{expt_guid}"
)

expt = collections.defaultdict(list)
for noise_var, result in zip(noise, results):
    cp = result.checkpoints[-1]
    expt["noise_var"].append(noise_var)
    expt["mean_reward_te"].append(cp.mean_reward_te)

print(pl.DataFrame(expt))
