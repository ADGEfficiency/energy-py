"""
"""

def run_single_episode(episode_number,
                       agent,
                       env,
                       sess=None):
    """
    Helper function to run through a single episode
    """

    #  initialize before starting episode
    done, step = False, 0
    observation = env.reset()
    #  while loop runs through a single episode
    while done is False:
        #  select an action
        action = agent.act(observation, sess)
        #  take one step through the environment
        next_observation, reward, done, info = env.step(action, episode_number)
        #  store the expWWerience
        agent.memory.add_experience(observation, action, reward, next_observation, step, episode_number)
        step += 1
        observation = next_observation

    #  now episode is done - process the episode in the agent memory
    agent.memory.process_episode(episode_number)
    return agent, env, sess

def train_model():
    pass

def output_results():
    pass
