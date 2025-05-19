import uuid

import numpy as np
import optuna
import polars as pl
from stable_baselines3 import PPO

import energypy
from energypy.dataset import load_electricity_prices

# Create a unique ID for this optimization run
optimization_id = str(uuid.uuid4())
log_dir = f"./data/tensorboard/bayesian_hpo/{optimization_id}"
print(f"logging to {log_dir}")

# Load data
data = load_electricity_prices()

# Create features: add some lags and horizons for price prediction
n_lags = 2
n_horizons = 6

data = data.with_columns(
    [pl.col("price").shift(n).alias(f"lag-{n}") for n in range(1, n_lags + 1)]
)
data = data.with_columns(
    [
        pl.col("price").shift(-1 * n).alias(f"horizon-{n}")
        for n in range(1, n_horizons + 1)
    ]
)
data = data.drop_nulls()

# Split data into train and test sets
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


# Define the objective function for Optuna
def objective(trial):
    """Optimization objective for Optuna."""

    # Create environments with the same data
    env_tr = energypy.make_env(electricity_prices=prices_tr, features=features_tr)
    env_te = energypy.make_env(electricity_prices=prices_te, features=features_te)

    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = 2000
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    n_epochs = 5
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # Neural network architecture
    net_arch_size = trial.suggest_int("net_arch_size", 16, 256, log=True)
    net_arch_layers = trial.suggest_int("net_arch_layers", 1, 3)
    net_arch = [net_arch_size] * net_arch_layers

    # Number of training steps
    # n_learning_steps = trial.suggest_int("n_learning_steps", 1000, 10000, log=True)
    n_learning_steps = 2000

    # Create a unique run ID for this trial
    run_guid = str(uuid.uuid4())

    # Create config with the suggested hyperparameters
    config = energypy.ExperimentConfig(
        env_tr=env_tr,
        env_te=env_te,
        agent=PPO(
            policy="MlpPolicy",
            env=env_tr,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=0,
            tensorboard_log=f"{log_dir}/runs/{run_guid}",
        ),
        name=f"hpo_trial_{trial.number}",
        n_learning_steps=n_learning_steps,
        n_eval_episodes=10,
    )

    # Run the experiment
    print(f"Running trial {trial.number} with params: {trial.params}")
    result = energypy.run_experiment(cfg=config)

    # Extract final test reward
    final_reward = result.checkpoints[-1].mean_reward_te

    # Optionally save the best model
    if trial.should_prune():
        raise optuna.TrialPruned()

    return final_reward


if __name__ == "__main__":
    # Create a study that maximizes the objective
    study = optuna.create_study(
        study_name="battery_optimization",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    # Run the optimization
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # Print optimization results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for param, value in best_trial.params.items():
        print(f"    {param}: {value}")

    # Plot optimization results
    try:
        # Importance of hyperparameters
        param_importance = optuna.visualization.plot_param_importances(study)
        param_importance.write_html(f"{log_dir}/param_importance.html")

        # Optimization history
        optimization_history = optuna.visualization.plot_optimization_history(study)
        optimization_history.write_html(f"{log_dir}/optimization_history.html")

        # Parallel coordinate plot
        parallel_plot = optuna.visualization.plot_parallel_coordinate(study)
        parallel_plot.write_html(f"{log_dir}/parallel_coordinate.html")

        print(f"Visualization plots saved to {log_dir}")
    except (ImportError, ModuleNotFoundError):
        print("Could not generate visualizations. Make sure plotly is installed.")

    # Train a final model with the best hyperparameters
    env_tr = energypy.make_env(electricity_prices=prices_tr, features=features_tr)
    env_te = energypy.make_env(electricity_prices=prices_te, features=features_te)

    best_params = best_trial.params

    # Extract neural network architecture
    net_arch_size = best_params.pop("net_arch_size")
    net_arch_layers = best_params.pop("net_arch_layers")
    net_arch = [net_arch_size] * net_arch_layers

    # Set training steps (using the same value as in the objective function)
    n_learning_steps = 2000  # This matches the hardcoded value in the objective function

    # Create final model config
    final_config = energypy.ExperimentConfig(
        env_tr=env_tr,
        env_te=env_te,
        agent=PPO(
            policy="MlpPolicy",
            env=env_tr,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            tensorboard_log=f"{log_dir}/final",
            **best_params,
        ),
        name="battery_best_hpo",
        n_learning_steps=n_learning_steps,
        n_eval_episodes=30,
    )

    # Train the final model
    print("Training final model with best hyperparameters:")
    final_result = energypy.run_experiment(cfg=final_config)

    print(f"Final model performance: {final_result.checkpoints[-1]}")
    print(f"Model saved to models/battery_best_hpo")

