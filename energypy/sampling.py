import numpy as np

from energypy import random_policy

from rich import print
from rich.progress import Progress
from energypy import utils


def episode(env, buffer, actor, hyp, counters, mode, return_info=False):
    obs = env.reset(mode=mode)
    done = False

    reward_scale = hyp["reward-scale"]

    #  hack for gym envs
    if isinstance(obs, np.ndarray):
        obs = {"features": obs}

    #  create one list per parallel episode we are running
    #  first dimension is the number of batteries
    #  which we use as the batch dimension when we are sampling actions from the agent
    episode_rewards = [list() for _ in range(obs["features"].shape[0])]

    infos = []
    while not done:
        #  obs is a dict {'features':, 'mask':}

        #  think I need to make obs a tensor here
        #  sometime returning np, sometimes returning a torch tensor
        #  I guess I really need my Actor to wrap around pytorch net
        act, _, deterministic_action = actor(obs["features"], obs["mask"])

        #  hack because our pytorch
        if not isinstance(act, np.ndarray):
            act = act.detach().numpy()
            deterministic_action = deterministic_action.detach().numpy()

        if mode == "test":
            act = deterministic_action

        #  next_obs is a dict {'next_obs', 'reward', 'done', 'next_obs_mask'}
        next_obs, reward, done, info = env.step(act)
        infos.append(info)

        #  want to save one observation per battery - buffer has no concept of batteries
        #  bit messy as I'm assuming the structure of the Transition tuple
        for i, (o, a, r, no, om, nom) in enumerate(
            zip(
                obs["features"],
                act,
                reward,
                next_obs["features"],
                obs["mask"],
                next_obs["mask"],
            )
        ):
            buffer.append(
                {
                    "observation": o,
                    "action": a,
                    "reward": r / reward_scale,
                    "next_observation": no,
                    "done": done,
                    "observation_mask": om,
                    "next_observation_mask": nom,
                }
            )
            episode_rewards[i].append(r)

        counters["env-steps"] += 1
        obs = next_obs

    episode_rewards = np.array(episode_rewards)
    if episode_rewards.ndim == 3:
        #  shape = (n_batteries, episode_len)
        episode_rewards = np.squeeze(episode_rewards, axis=2)
        #  shape = (n_batteries, )
        episode_rewards = episode_rewards.sum(axis=1)

    else:
        assert episode_rewards.ndim == 2
        episode_rewards = episode_rewards.sum(axis=1)

    if return_info:
        return episode_rewards, infos
    else:
        return episode_rewards


def run_episode(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    mode,
):
    st = utils.now()
    episode_rewards = episode(
        env,
        buffer,
        actor,
        hyp,
        counters,
        mode,
    )

    #  this should be a func
    for episode_reward in episode_rewards:
        episode_reward = float(episode_reward)

        #  so much repetition TODO
        rewards["episode-reward"].append(episode_reward)
        rewards[f"{mode}-reward"].append(episode_reward)

        writers[mode].scalar(
            episode_reward,
            f"{mode}-episode-reward",
            f"{mode}-episodes",
        )
        writers["episodes"].scalar(episode_reward, "episode-reward", "episodes")

        writers[mode].scalar(
            utils.last_100_episode_rewards(rewards[f"{mode}-reward"]),
            f"last-100-{mode}-rewards",
            f"{mode}-episodes",
        )

        writers["episodes"].scalar(
            utils.last_100_episode_rewards(rewards[f"episode-reward"]),
            "last-100-episode-rewards",
            "episodes",
        )

        counters["episodes"] += 1
        counters[f"{mode}-episodes"] += 1

    counters["sample-seconds"] += utils.now() - st
    counters[f"sample-{mode}-seconds"] += utils.now() - st
    return episode_rewards


def sample_random(
    env,
    buffer,
    hyp,
    writers,
    counters,
    rewards,
):
    mode = "random"
    print(f" filling buffer with {buffer.size} samples")
    policy = random_policy.make(env)

    while not buffer.full:
        run_episode(
            env,
            buffer,
            policy,
            hyp,
            writers,
            counters,
            rewards,
            mode,
        )

    assert len(buffer) == buffer.size
    print(f" buffer filled with {len(buffer)} samples\n")
    return buffer


def sample_test(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
):
    env.setup_test(hyp["n-tests"])
    n_test_eps = env.n_test_eps
    print(f" testing on {n_test_eps} episodes")

    #  this will need updating for the battery env stuff
    # try:
    #     n_test_eps = len(env.dataset.episodes["test"])

    # #  env without dataset - we fall back on hyperparameters
    # except AttributeError:
    #     n_test_eps = hyp["n-tests"]

    test_results = []
    test_done = env.test_done
    with Progress() as progress:
        task = progress.add_task("Running test episode...", total=n_test_eps)

        while not test_done:
            test_rewards = run_episode(
                env,
                buffer,
                actor,
                hyp,
                writers,
                counters,
                rewards,
                mode="test",
            )
            test_results.extend(test_rewards)
            test_done = env.test_done
            progress.update(task, advance=len(test_results))

    utils.stats("test", "test-episodes", counters, test_rewards)
    return test_results


def sample_train(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
):
    episode_rewards = run_episode(
        env, buffer, actor, hyp, writers, counters, rewards, mode="train"
    )
    utils.stats("train", "train-episodes", counters, episode_rewards)
    return episode_rewards
