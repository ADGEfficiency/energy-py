from energy_py import expt_args, save_args, make_logger, make_paths
from energy_py import EternityVisualizer
from energy_py.agents import DQN, KerasQ
from energy_py.envs import FlexEnv


def dqn_experiment(env, data_path, base_path='dqn_agent'):
    parser, args = expt_args({'name': '--bs',
                              'type': int,
                              'default': 64,
                              'help': 'batch size for experience replay'})
    EPISODES = args.ep
    EPISODE_LENGTH = args.len
    BATCH_SIZE = args.bs
    DISCOUNT = args.gamma
    OUTPUT_RESULTS = args.out
    LOAD_BRAIN = False

    paths = make_paths(base_path)
    BRAIN_PATH = paths['brain']
    RESULTS_PATH = paths['results']
    ARGS_PATH = paths['args']
    LOG_PATH = paths['logs']

    logger = make_logger(LOG_PATH)

    env = env(data_path, episode_length=EPISODE_LENGTH)

    #  total steps is used to setup hyperparameters for the DQN agent
    total_steps = EPISODES * env.observation_ts.shape[0]

    agent = DQN(env, 
                DISCOUNT, 
                brain_path=BRAIN_PATH,
                Q=KerasQ,
                batch_size=BATCH_SIZE,
                total_steps=total_steps)

    save_args(args, 
              path=ARGS_PATH,
              optional={'total steps': total_steps,
                        'epsilon decay steps': agent.epsilon_decay_steps,
                        'update target net': agent.update_target_net,
                        'memory length': agent.memory.length,
                        'load brain': LOAD_BRAIN, 
                        'initial random': agent.initial_random}) 

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


            if step > agent.initial_random:

                #  with DQN we can learn within episode 
                #  get a batch to learn from
                obs, acts, rews, next_obs = agent.memory.get_random_batch(BATCH_SIZE)

                loss = agent.learn(observations=obs,
                                   actions=acts,
                                   rewards=rews,
                                   next_observations=next_obs)
            
            if step % agent.update_target_net == 0:
                agent.update_target_network()

        if episode % OUTPUT_RESULTS == 0:
            #  collect data from the agent & environment
            hist = EternityVisualizer(agent,
                                      env,
                                      results_path=RESULTS_PATH)

            agent_outputs, env_outputs = hist.output_results(save_data=True)

if __name__ == '__main__':
    env = FlexEnv
    agent_outputs, env_outputs = dqn_experiment(env)
