import tensorflow as tf

from energy_py import experiment, save_args, Timer
from energy_py import EternityVisualizer
from energy_py.agents import Q_DQN 


@experiment()
def experiment(agent, args, paths, env, opt_agent_args=[]):
    timer = Timer()

    paths = make_paths(base_path)

    logger = make_logger(paths['logs'], args.log)

    env = env(data_path,
              episode_length=args.len,
              episode_random=args.rand)


    ARGS_PATH = paths['args']

    total_steps = config['total_steps']
    agent = DQN(**config)

    save_args(args, path=ARGS_PATH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step, episode = 0, 0

        #  outer while loop runs through multiple episodes
        while step < total_steps:
            episode += 1
            done, step = False, 0
            observation = env.reset()

            #  while loop runs through a single episode
            while done is False:
                step += 1

                #  select an action
                action = agent.act(sess=sess, obs=observation)
                #  take one step through the environment
                next_observation, reward, done, info = env.step(action)
                #  store the experience
                agent.add_experience(observation, action, reward,
                                            next_observation, done)
                observation = next_observation

                if step > agent.initial_random:
                    train_info = agent.learn()

            #  reporting expt status at the end of each episode
            timer.report({'episode': episode,
                          'ep start': env.state_ts.index[0],
                          'lifetime avg rew': agent.memory.rewards.mean()})

    return train_info
