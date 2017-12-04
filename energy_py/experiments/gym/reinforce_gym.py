import gym

from energy_py import expt_args, save_args, make_logger, make_paths 
from energy_py import EternityVisualizer

from energy_py.agents import REINFORCE, GaussianPolicy

env = gym.envs.make('MountainCarContinuous-v0')
#env = gym.envs.make('Pendulum-v0')

def gym_experiment(agent, env, data_path, base_path='gym_expt'):
    parser, args = expt_args()
    EPISODES = args.ep
    DISCOUNT = args.gamma
    LEARNING_RATE = args.lr
    OUTPUT_RESULTS = args.out

    paths = make_paths(base_path)

    paths = make_paths(name)
    BRAIN_PATH = paths['brain']
    RESULTS_PATH = paths['results']
    LOG_PATH = paths['logs']

    logger = make_logger(LOG_PATH)

    env = env(data_path, episode_length=EPISODE_LENGTH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(1, EPISODES):
            state = env.reset()
            state = state.reshape(-1, env.observation_space.shape[0]) 

            done, step = False, 0
            rewards = []
            while not done:
                env.render()    
                action = agent.act(observation=state,
                                   session=sess)
                next_state, reward, done, info = env.step(action)
                rewards.append(reward)
                agent.memory.add_experience(state, action, reward, next_state, step, episode)

                step += 1
                state = next_state
            print('ep {} total reward {} max {}'.format(episode, 
                                                        sum(rewards),
                                                        max(rewards)))
            #  now episode is over we can learn
            
            obs, acts, rews = agent.memory.get_episode_batch(episode)

            loss = agent.learn(observations=obs,
                               actions=acts,
                               rewards=rews,
                               session=sess)

            total_rew = np.sum(rews)
            agent.memory.info['total ep rewards'].append(total_rew)

            def print_stats(total_rew):
                print('total {}'.format(np.sum(total_rew)))
                print('mean {}'.format(np.mean(total_rew)))
                print('std {} '.format(np.std(total_rew)))

            global_history = EternityVisualizer(episode, agent, env=None, results_path=RESULTS_PATH) 
            outputs = global_history.output_results(save_data=False)
