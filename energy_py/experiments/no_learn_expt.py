from energy_py import experiment, save_args, EternityVisualizer


@experiment()
def no_learning_experiment(agent, args, paths, env):
    """
    Runs an experiment with an agent that doesn't need to learn
    """
    agent = agent(env, args.gamma)

    save_args(args, paths['args'])

    for episode in range(1, args.ep):

        #  initialize before starting episode
        done, step = False, 0
        observation = env.reset(episode)

        #  while loop runs through a single episode
        while done is False:
            #  select an action
            action = agent.act(observation=observation)
            #  take one step through the environment
            next_observation, reward, done, info = env.step(action)
            #  store the experience
            agent.memory.add_experience(observation, action, reward,
                                        next_observation, done, step, episode)
            step += 1
            observation = next_observation

        if episode % args.out == 0:
            #  collect data from the agent & environment
            hist = EternityVisualizer(agent,
                                      env,
                                      paths['results'])

            agent_outputs, env_outputs = hist.output_results(save_data=True)

    return agent_outputs, env_outputs
