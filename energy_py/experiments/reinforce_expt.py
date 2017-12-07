import tensorflow as tf

from energy_py import expt_args, save_args, make_logger, make_paths, run_single_episode
from energy_py import EternityVisualizer
from energy_py.agents import REINFORCE, GaussianPolicy
from energy_py.envs import FlexEnv

def reinforce_experiment(env, data_path, base_path='reinforce_agent'):
    parser, args = expt_args()

    EPISODES = args.ep
    EPISODE_LENGTH = args.len
    DISCOUNT = args.gamma
    OUTPUT_RESULTS = args.out
    LEARNING_RATE = 0.0001

    paths = make_paths(base_path)
    BRAIN_PATH = paths['brain']
    RESULTS_PATH = paths['results']
    ARGS_PATH = paths['args']
    LOG_PATH = paths['logs']

    logger = make_logger(LOG_PATH)

    env = env(data_path, episode_length=EPISODE_LENGTH)

    #  total steps is used to setup hyperparameters for the DQN agent
    total_steps = EPISODES * env.observation_ts.shape[0]

    agent = REINFORCE(env, 
                DISCOUNT, 
                BRAIN_PATH,
                policy=GaussianPolicy,
                lr=LEARNING_RATE)

    save_args(args, 
              path=ARGS_PATH,
              optional={'total steps': total_steps,
                        'learning rate': LEARNING_RATE})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(1, EPISODES):

            #  initialize before starting episode
            done, step = False, 0
            observation = env.reset(episode)

            #  while loop runs through a single episode
            while done is False:
                agent, env, sess = run_single_episode(episode,
                                                      agent,
                                                      env,
                                                      sess)
                #  now experiment is over, we can learn
                obs, acts, rews = agent.memory.get_episode_batch(episode)
                returns = agent.calculate_returns(rews)

                loss = agent.learn(observations=obs,
                                   actions=acts,
                                   discounted_returns=returns,
                                   session=sess)

            if episode % OUTPUT_RESULTS == 0:
                #  collect data from the agent & environment
                hist = EternityVisualizer(agent,
                                          env,
                                          results_path=RESULTS_PATH)

                agent_outputs, env_outputs = hist.output_results(save_data=True)
    return agent_outputs, env_outputs

if __name__ == '__main__':
    env = FlexEnv
    agent_outputs, env_outputs = reinforce_experiment(env)
