import logging
import os

import numpy as np

from energy_py import Normalizer, Standardizer
from energy_py.agents import BaseAgent, EpsilonGreedy

logger = logging.getLogger(__name__)


class DQN(BaseAgent):
    """
    energy_py implementation of DQN
    aka Q-learning with experience replay & target network

    args
        env                 : energy_py environment
        Q                   : energy_py Action-Value Function Q(s,a)
        discount
        batch_size

    inherits from
        Base_Agent          : the energy_py class used for agents

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self, 
                 env,
                 discount,

                 Q,
                 batch_size,
                 total_steps,

                 brain_path=[],
                 load_agent_brain=False):
        """
        DeepMind Atari hyperparameters are here as an FYI 

        batch_size = 32
        replay_memory_size = 1 million
        target network update freq = 10000
        replay start size = 50000 (steps of exp used to populate memory)
        learning rate = 0.00025
        total frames = 10 million
        """
        memory_length = int(total_steps * 0.1)
        self.epsilon_decay_steps = int(total_steps / 2)
        self.update_target_net = int(total_steps * 0.0125)
        self.initial_random = int(total_steps * 0.1)

        #  passing the environment to the BaseAgent class
        super().__init__(env, discount, brain_path, memory_length=memory_length)

        self.state_processor = Standardizer(length=self.observation_space.shape[0])
        self.target_processor = Normalizer(length=self.action_space.shape[0])

        self.actions = self.action_space.discretize(num_discrete=10)

        model_dict = {'input_nodes': self.observation_space.shape[0],
                      'output_nodes': self.actions.shape[0],
                      'layers'    : [25, 25],
                      'lr'        : 0.0025,
                      'batch_size': batch_size,
                      'epochs'    : 1}

        #  make our two action value functions
        self.Q_actor = Q(model_dict, 'Q_actor')
        self.Q_target = Q(model_dict, 'Q_target')

        #  create an object to decay epsilon
        self.e_greedy = EpsilonGreedy(self.initial_random,
                                      self.epsilon_decay_steps)

        if load_agent_brain:
            self.load_brain()

    def _reset(self):
        """
        Resets the agent
        """
        self.Q_actor.model.reset_weights()
        self.Q_target.model.reset_weights()
        self.e_greedy.reset()

    def _act(self, **kwargs):
        """
        Act using an epsilon-greedy policy

        args
            sess (tf.Session): the current tensorflow session
            obs (np.array): shape=(1, observation_space.shape[0])

        return
            action      : np array (1, action_space.shape[0])
        """
        sess = kwargs.pop('sess')
        observation = kwargs.pop('obs')

        #  using our state processor to transform the observation 
        observation = self.state_processor.transform(observation)

        #  get the current value of epsilon
        epsilon = self.e_greedy.epsilon
        self.memory.info['epsilon'].append(epsilon)

        if np.random.uniform() < epsilon:
            #  acting randomly
            logger.debug('epsilon {:.3f} - acting randomly'.format(epsilon))
            action = self.action_space.sample() 

        else:
            #  get predictions from the action_value function Q
            Q_estimates = self.Q_actor.predict(sess, observation)

            #  select the action with the highest Q
            action = self.actions[np.argmax(Q_estimates)]

            #  calculate some statistics from the Q estimates
            max_Q = np.max(Q_estimates)
            avg_Q = np.mean(Q_estimates)

            #  save some data for debugging later
            self.memory.info['max_Q_acting estimates'].append(max_Q)
            self.memory.info['avg_Q_acting estimates'].append(avg_Q)
            self.memory.info['Q act est'].extend(Q_estimates.flatten().tolist())

            logger.debug('using Q_actor - max(Q_est)={:.3f}'.format(max_Q))
            logger.debug('using Q_actor - avg(Q_est)={:.3f}'.format(avg_Q))

        #  make sure action is shaped correctly
        action = np.array(action).reshape(1, self.action_space.shape[0])
        assert self.action_space.shape[0] == action.shape[1]

        return action

    def _learn(self, **kwargs):
        """
        Update Q_actor using the Bellman Equation

        observations, actions, rewards should all be either
        normalized or standardized

        args
            sess (tf.Session)

            batch (dict) keys=str, val=np.array 
                obs (batch_size, observation_space.shape[0])
                acts (batch_size, num_actions)
                rews (batch_size, 1)
                next_obs (batch_size, observataion_dim)
                terminal (bool)

        returns
            history             : list
        """
        sess = kwargs.pop('sess')

        #  grab the batch dictionary and pull out data
        batch = kwargs.pop('batch')
        observations = batch['obs']
        actions = batch['acts']
        rewards = batch['rews']
        next_observations = batch['next_obs']
        terminal = batch['terminal']

        #  check that we have equal number of all of our inputs
        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == rewards.shape[0]
        assert observations.shape[0] == next_observations.shape[0]
        assert observations.shape[0] == terminal.shape[0]

        #  we process the entire batch of inputs using our state_processor
        inputs = self.state_processor.transform(observations)
        targets = np.zeros(shape=(observations.shape[0], 
                                  self.Q_actor.output_nodes))
        assert inputs.shape[0] == targets.shape[0]

        #  iterate over the experience to create the input and target
        for j, (rew, next_obs, term) in enumerate(zip(rewards,
                                                      next_observations,
                                                      terminal)):
            if term:
                #  if the next state is terminal
                #  the return of our current state is equal to the reward
                #  i.e. Q(s',a) = 0 for any a
                target = rew

            else:
                #  if not terminal then we need to predict the max return
                #  from the next_observation
                #  note that we use Q_target here
                next_obs = self.state_processor.transform(next_obs)
                max_q = np.max(self.Q_target.predict(sess, next_obs))

                #  can now use Bellman equation to create a target
                #  using the max value of s'
                target = rew + self.discount * max_q

            #  save the network inputs and targets  
            targets[j] = target

        logger.debug('Finished iterating over batch of experience')
        #  save the unscaled targets so we can visualize later
        self.memory.info['unscaled Q targets'].extend(list(targets.flatten()))

        #  scaling the targets by normalizing
        targets = self.target_processor.transform(targets)
        #  reshape targets into 2 dimensions
        targets = targets.reshape(-1, self.Q_actor.output_nodes)

        #  update our Q function
        assert inputs.shape[0] == targets.shape[0]
        error, loss = self.Q_actor.improve(sess, inputs, targets)

        #  save loss and the training targets for visualization later
        self.memory.info['train error'].extend(error.flatten().tolist())
        self.memory.info['loss'].append(loss)

        self.memory.info['train Q inputs'].extend(inputs.flatten().tolist())
        self.memory.info['train Q targets'].extend(targets.flatten().tolist())

        #  save some data for analysis later 
        max_target = np.max(targets)
        avg_target = np.mean(targets)

        self.memory.info['max learning target'].append(max_target)
        self.memory.info['avg learning target'].append(avg_target)
        self.memory.info['learning targets'].extend(targets.flatten().tolist())
        
        logger.debug('learning - max(Q_est)={:.3f}'.format(max_target))
        logger.debug('learning - avg(Q_est)={:.3f}'.format(avg_target))

        return error, loss 

    def _load_brain(self):
        """
        Loads experiences, Q_actor and Q_target
        """
        #  load the epsilon greedy object
        e_greedy_path = os.path.join(self.brain_path, 'e_greedy.pickle') 
        self.e_greedy = self.load_pickle(e_greedy_path)

        #  load the action value functions
        Q_actor_path = os.path.join(self.brain_path, 'Q_actor.h5')
        self.Q_actor.load_model(Q_actor_path)
        self.Q_target = self.Q_actor

    def _save_brain(self):
        """
        Saves experiences, Q_actor and Q_target
        """
        e_greedy_path = os.path.join(self.brain_path, 'e_greedy.pickle') 
        self.dump_pickle(self.e_greedy, e_greedy_path)

        """
        add the acting Q network
        we don't add the target network - we just use the acting network
        to initialize Q_target when we load_brain
        not reccomended to use pickle for Keras models
        so we use h5py to save Keras models
        """ 
        Q_actor_path = os.path.join(self.brain_path, 'Q_actor.h5')
        self.Q_actor.save_model(Q_actor_path)

    def update_target_network(self, sess):
        """
        Copies weights from Q_actor into Q_target
        """
        logger.info('Updating Q_target by copying weights from Q_actor')
        sess = self.Q_target.copy_weights(sess, parent=self.Q_actor)
        return sess
