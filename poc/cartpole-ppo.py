import gymnasium as gym
from stable_baselines3 import PPO

from energypy.runner import main

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    main(
        env=env,
        eval_env=gym.make("CartPole-v1"),
        model=PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
        ),
        name="cartpole",
    )
