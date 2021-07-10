import numpy as np

from energypy import random_policy

from tqdm import tqdm
from energypy import utils


def episode(
    env,
    buffer,
    actor,
    hyp,
    counters,
    mode,
    logger=None
):
    obs = env.reset(mode=mode)
    done = False

    reward_scale = hyp['reward-scale']

    #  create one list per parallel episode we are running
    episode_rewards = [list() for _ in range(obs.shape[0])]

    while not done:
        act, _, deterministic_action = actor(obs)

        if mode == 'test':
            act = deterministic_action

        next_obs, reward, done, _ = env.step(np.array(act))

        for i, (o, a, r, no) in enumerate(zip(obs, act, reward, next_obs)):
            buffer.append(env.Transition(o, a, r/reward_scale, no, done))
            episode_rewards[i].append(r)

        if logger:
            logger.debug(
                f'{obs}, {act}, {reward}, {next_obs}, {done}, {mode}'
            )

        counters['env-steps'] += 1
        obs = next_obs

    episode_rewards = np.array(episode_rewards)
    if episode_rewards.ndim == 3:
        #  shape = (n_batteries, episode_len)
        episode_rewards = np.squeeze(episode_rewards, axis=2)
        #  shape = (n_batteries, )
        return episode_rewards.sum(axis=1)

    else:
        assert episode_rewards.ndim == 2
        return episode_rewards.sum(axis=1)


def run_episode(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    mode,
    logger=None
):
    st = utils.now()
    episode_rewards = episode(
        env,
        buffer,
        actor,
        hyp,
        counters,
        mode,
        logger=logger,
    )

    for episode_reward in episode_rewards:
        episode_reward = float(episode_reward)

        #  so much repetition TODO
        rewards['episode-reward'].append(episode_reward)
        rewards[f'{mode}-reward'].append(episode_reward)

        writers[mode].scalar(
            episode_reward,
            f'{mode}-episode-reward',
            f'{mode}-episodes',
        )
        writers['episodes'].scalar(
            episode_reward,
            'episode-reward',
            'episodes'
        )

        writers[mode].scalar(
            utils.last_100_episode_rewards(rewards[f'{mode}-reward']),
            f'last-100-{mode}-rewards',
            f'{mode}-episodes',
        )

        writers['episodes'].scalar(
            utils.last_100_episode_rewards(rewards[f'episode-reward']),
            'last-100-episode-rewards',
            'episodes',
        )

        counters['episodes'] += 1
        counters[f'{mode}-episodes'] += 1

    counters['sample-seconds'] += utils.now() - st
    counters[f'sample-{mode}-seconds'] += utils.now() - st
    return episode_rewards


def sample_random(
    env,
    buffer,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'random'
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
            logger=logger
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
    logger,
):
    env.setup_test()

    test_results = []
    test_done = False
    try:
        n_test_eps = len(env.dataset.episodes['test'])

    #  TODO - env without dataset
    except AttributeError:
        n_test_eps = hyp['n-tests']

    print(f' testing on {n_test_eps} episodes')
    pbar = tqdm(total=n_test_eps)
    while not test_done:

        test_rewards = run_episode(
            env,
            buffer,
            actor,
            hyp,
            writers,
            counters,
            rewards,
            mode='test',
            logger=logger
        )
        test_results.extend(test_rewards)
        test_done = env.test_done
        pbar.update(len(test_results))

    pbar.close()
    utils.stats('test', 'test-episodes', counters, test_rewards)
    return test_results


def sample_train(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    episode_rewards = run_episode(
        env,
        buffer,
        actor,
        hyp,
        writers,
        counters,
        rewards,
        mode='train',
        logger=logger
    )
    utils.stats('train', 'train-episodes', counters, episode_rewards)
    return episode_rewards
