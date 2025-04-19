import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np


def main(env, eval_env, model, name):
    # 3. Train the model
    model.learn(total_timesteps=50000)

    # 4. Save the trained model
    model.save(f"models/{name}")

    # 5. Load the trained model (optional)
    # model = PPO.load("ppo_cartpole")

    # 6. Create a manual interaction loop
    def interact_with_environment(env, model, num_episodes=5):
        """Interact with the environment using the trained model and display results."""
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done:
                # Act: Get the action from the model
                # TODO - should deterministic only be when in test mode?
                action, _states = model.predict(obs, deterministic=True)

                # Step: Execute the action in the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Observe reward
                total_reward += reward

                # Print current state
                print(f"Episode {episode + 1}, Step {step_counter + 1}")
                print(f"  Observation: {obs}")
                print(f"  Action: {action}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")
                print("---")

                # Update observation
                obs = next_obs
                step_counter += 1

            print(
                f"Episode {episode + 1} completed with total reward: {total_reward}, steps: {step_counter}"
            )
            print("=" * 50)

    # 7. Run the interaction loop
    interact_with_environment(eval_env, model)

    # 8. Evaluate the model more formally
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # # 9. Record a video of the trained agent using SB3's built-in recorder
    # from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

    # def record_video(model, env_id, video_folder="videos", video_length=500):
    #     """
    #     Record a video of an agent's performance.

    #     Args:
    #         model: The trained model
    #         env_id: The environment ID (string)
    #         video_folder: Where to save the video
    #         video_length: Length of recording in timesteps

    #     Returns:
    #         Path to the video folder
    #     """

    #     # Create a vectorized environment for recording
    #     def make_env():
    #         return gym.make(env_id, render_mode="rgb_array")

    #     vec_env = DummyVecEnv([make_env])

    #     # Create the recorder
    #     video_env = VecVideoRecorder(
    #         vec_env,
    #         video_folder,
    #         record_video_trigger=lambda x: x == 0,  # Record at the beginning
    #         video_length=video_length,
    #         name_prefix=f"ppo-{env_id}",
    #     )

    #     # Reset the environment
    #     obs = video_env.reset()

    #     # Run for video_length steps or until done
    #     for _ in range(video_length):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, _, dones, _ = video_env.step(action)
    #         if dones.any():
    #             # If the episode ends, reset
    #             obs = video_env.reset()

    #     # Close the environment
    #     video_env.close()

    #     return video_folder

    # # Record a video of the trained agent
    # video_path = record_video(model, "CartPole-v1")
    # print(f"Video saved to {video_path}")

    # # 9. Clean up
    # env.close()
    # eval_env.close()
