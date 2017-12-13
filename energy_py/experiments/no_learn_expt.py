from energy_py import expt_args, save_args, make_logger, make_paths, EternityVisualizer


def no_learning_experiment(agent, env, data_path, base_path='no_learn_agent'):
    """
    Runs an experiment with an agent that doesn't need to learn
    """
    parser, args = expt_args()
    EPISODES = args.ep
    EPISODE_LENGTH = args.len
    DISCOUNT = args.gamma
    OUTPUT_RESULTS = args.out
    LOG_STATUS = args.log

    paths = make_paths(base_path)
    BRAIN_PATH = paths['brain']
    RESULTS_PATH = paths['results']
    ARGS_PATH = paths['args']
    LOG_PATH = paths['logs']

    logger = make_logger(LOG_PATH, LOG_STATUS)

    save_args(args, path=ARGS_PATH) 

    env = env(data_path, episode_length=EPISODE_LENGTH)

    agent = agent(env, DISCOUNT)

    for episode in range(1, EPISODES):

        #  initialize before starting episode
        done, step = False, 0
        observation = env.reset(episode)

        #  while loop runs through a single episode
        while done is False:
            #  we cheat a little bit here by grabbing the observation as a
            #  pandas series
            #  just want to pull out the timestamp
            time = env.observation_ts.index[env.steps]
            print('time stamp is {}'.format(time))
            #  select an action
            action = agent.act(observation=observation, timestamp=time)
            #  take one step through the environment
            next_observation, reward, done, info = env.step(action)
            #  store the experience
            agent.memory.add_experience(observation, action, reward,
                                        next_observation, done, step, episode)
            step += 1
            observation = next_observation

        if episode % OUTPUT_RESULTS == 0:
            #  collect data from the agent & environment
            hist = EternityVisualizer(agent,
                                      env,
                                      results_path=RESULTS_PATH)

            agent_outputs, env_outputs = hist.output_results(save_data=True)
