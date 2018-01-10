import tensorflow as tf

from energy_py import experiment, save_args
from energy_py import EternityVisualizer
from energy_py.agents import Q_DQN 


@experiment()
def dqn_experiment(agent, args, paths, env, opt_agent_args=[]):
#  not using opt agent args here
    EPISODES = args.ep 
    DISCOUNT = args.gamma 
    BATCH_SIZE = args.bs
    OUTPUT_FREQ = args.out

    ARGS_PATH = paths['args']
    RESULTS_PATH = paths['results']

    #  total steps is used to setup hyperparameters for the DQN agent
    total_steps = EPISODES * env.observation_ts.shape[0]

    agent = agent(env, 
                  DISCOUNT, 
                  Q=Q_DQN,
                  total_steps=total_steps)

    save_args(args, 
              path=ARGS_PATH,
              optional={'total steps': total_steps,
                        'epsilon decay (steps)': agent.epsilon_decay_steps,
                        'update target net (steps)': agent.update_target_net,
                        'memory length (steps)': agent.memory.memory_length,
                        'initial random (steps)': agent.initial_random}) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_step = 0

        for episode in range(1, EPISODES):

            #  initialize before starting episode
            done, step = False, 0
            observation = env.reset()

            #  while loop runs through a single episode
            while done is False:
                #  select an action
                action = agent.act(sess=sess, obs=observation)
                #  take one step through the environment
                next_observation, reward, done, info = env.step(action)
                #  store the experience
                agent.memory.add_experience(observation, action, reward,
                                            next_observation, done, step, episode)
                step += 1
                total_step += 1
                observation = next_observation

                if total_step > agent.initial_random:

                    #  with DQN we can learn within episode 
                    #  get a batch to learn from
                    batch = agent.memory.get_random_batch(BATCH_SIZE)

                    train_info = agent.learn(sess=sess, batch=batch)
                        
                    if total_step % agent.update_target_net == 0:
                        agent.update_target_network(sess)

            if episode % OUTPUT_FREQ == 0:
                #  collect data from the agent & environment
                hist = EternityVisualizer(agent,
                                          env,
                                          results_path=RESULTS_PATH)

                agent_outputs, env_outputs = hist.output_results(save_data=True)

        hist = EternityVisualizer(agent,
                                  env,
                                  results_path=RESULTS_PATH)

        agent_outputs, env_outputs = hist.output_results(save_data=True)

    return agent_outputs, env_outputs
