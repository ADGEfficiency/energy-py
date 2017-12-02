from energy_py import expt_args, save_args, make_logger, make_paths, EternityVisualizer
from energy_py.agents import RandomAgent 
from energy_py.envs import FlexEnv


def random_experiment(env, data_path, base_path='random_agent'):
    parser, args = expt_args()
    EPISODES = args.ep
    EPISODE_LENGTH = args.len
    DISCOUNT = args.gamma
    OUTPUT_RESULTS = args.out

    paths = make_paths('random_agent')
    BRAIN_PATH = paths['brain']
    RESULTS_PATH = paths['results']
    ARGS_PATH = paths['args']
    LOG_PATH = paths['logs']

    logger = make_logger(LOG_PATH)

    save_args(args, path=ARGS_PATH) 

    env = env(data_path, episode_length=EPISODE_LENGTH)

    agent = RandomAgent(env, DISCOUNT)

    for episode in range(1, EPISODES):

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
                                        next_observation, step, episode)
            step += 1
            observation = next_observation

        if episode % OUTPUT_RESULTS == 0:
            #  collect data from the agent & environment
            hist = EternityVisualizer(agent,
                                      env,
                                      results_path=RESULTS_PATH)

            agent_outputs, env_outputs = hist.output_results(save_data=True)

if __name__ == 'main':
    env = FlexEnv
    agent_outputs, env_outputs = random_experiment(env)
