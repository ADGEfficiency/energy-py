import logging

import numpy as np
import tensorflow as tf

from energy_py.agents import BaseAgent


class REINFORCE(BaseAgent):
    """
    Monte Carlo implementation of REINFORCE
    No baseline - true Monte Carlo returns used

    args
        env (energy_py environment)
        discount (float) 
        brain_path (str) : directory where brain lives
        policy (energy_py policy approximator)
        learning rate (float)

    Monte Carlo REINFORCE is high variance and low bias
    Variance can be reduced through the use of a baseline TODO 

    This algorithm requires lots of episodes to run:
    - policy gradient only makes small updates
    - Monte Carlo is high variance (takes a while for expectation to converge)
    - we only update once per episode
    - only learn from samples once

    Reference = Williams (1992)
    """
    def __init__(self,
                 env,
                 discount,
                 brain_path,

                 policy,
                 lr,
                 process_reward,
                 process_return):

        super().__init__(env, discount, brain_path,
                         process_reward, process_return)
        
        #  create the policy function approximator
        self.policy = policy(num_actions=self.num_actions,
                             observation_dim=self.observation_dim, 
                             lr=lr,
                             action_space=self.action_space)

    def _act(self, **kwargs):
        """
        Act according to the policy network

        args
            observation : np array (1, observation_dim)
            session     : a TensorFlow Session object

        return
            action      : np array (1, num_actions)
        """
        observation = kwargs.pop('observation')
        session = kwargs.pop('session')

        #  scaling the observation for use in the policy network
        scaled_observation = self.memory.scale_array(observation, self.observation_space)

        scaled_observation = scaled_observation.reshape(-1, self.observation_dim)
        assert scaled_observation.shape[0] == 1

        #  generating an action from the policy network
        action, output = self.policy.get_action(session, scaled_observation)

        for i, mean in enumerate(output['means'].flatten()):
            self.memory.info['mean {}'.format(i)].append(mean)

        logging.debug('scaled_obs {}'.format(scaled_observation))
        logging.debug('action {}'.format(action))

        self.memory.info['scaled_obs'].extend(list(scaled_observation.flatten()))
        self.memory.info['action'].extend(list(action.flatten()))

        logging.debug('means are {}'.format(output['means']))
        logging.debug('stdevs are {}'.format(output['stdevs']))

        return action.reshape(-1, self.num_actions)

    def _learn(self, **kwargs):
        """
        Update the policy network using the episode experience

        args
            observations        : np array (episode_length, observation_dim)
            actions             : np array (episode_length, num_actions)
            discounted_returns  : np array (episode_length, 1)
            session             : a TensorFlow Session object

        return
            loss                : np float
        """
        observations = kwargs.pop('observations')
        actions = kwargs.pop('actions')
        discounted_returns = kwargs.pop('discounted_returns')
        session = kwargs.pop('session')

        logging.debug('observations {}'.format(observations))
        logging.debug('actions {}'.format(actions))
        logging.debug('discounted_returns {}'.format(discounted_returns))

        loss = self.policy.improve(session,
                                   observations,
                                   actions,
                                   discounted_returns)

        self.memory.info['losses'].append(loss)
        logging.info('loss is {:.8f}'.format(loss))

        return loss
