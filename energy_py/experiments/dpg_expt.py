import tensorflow as tf

from energy_py import experiment, save_args
from energy_py import EternityVisualizer
from energy_py.agents import DPGActor, DPGCritic, OrnsteinUhlenbeckActionNoise


@experiment()
def dpg_experiment(agent, args, paths, env):

    EPISODES = args.ep 
    DISCOUNT = args.gamma 
    OUTPUT_FREQ = args.out
    BATCH_SIZE = 64

    ARGS_PATH = paths['args']
    RESULTS_PATH = paths['results']

    agent = agent(env, 
                  DISCOUNT,
                  DPGActor,
                  DPGCritic)
    agent.initial_random = 10
    agent.update_target_net = 10
    save_args(args, path=ARGS_PATH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_step = 0

        for episode in range(1, EPISODES):

            #  initialize before starting episode
            done, step = False, 0
            observation = env.reset(episode)

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
                        agent.actor.update_target_net()
                        agent.critic.update_target_net()

            if episode % OUTPUT_FREQ == 0:
                #  collect data from the agent & environment
                hist = EternityVisualizer(agent,
                                          env,
                                          results_path=RESULTS_PATH)

                agent_outputs, env_outputs = hist.output_results(save_data=True)

    return agent_outputs, env_outputs
