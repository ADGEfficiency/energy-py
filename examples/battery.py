import numpy as np
from stable_baselines3 import PPO

import energypy

env = energypy.make_env(electricity_prices=np.random.uniform(-1000, 1000, 1024 * 5))
config_random = energypy.ExperimentConfig(
    env_tr=env,
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
        verbose=1,
    ),
    name="battery_random",
    n_eval_episodes=5,
)

result = energypy.run_experiment(cfg=config_random)
print(f"Random price model performance: {result.checkpoints[-1]}")
