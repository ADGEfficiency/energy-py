import logging

import numpy as np

from energy_py import Normalizer, Standardizer
from energy_py.agents import BaseAgent, EpsilonGreedy

logger = logging.getLogger(__name__)


class DQN(BaseAgent):
    """
    energy_py implementation of DQN
    aka Q-learning with experience replay & target network

    args
        env (object) energy_py environment
        Q (object) energy_py Action-Value Function Q(s,a)
        discount (float) discount rate aka gamma
        total_steps (int) number of steps in agent life
        discrete_space_size (int) number of discrete spaces

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self,
                 env,
                 discount,
                 Q,
                 total_steps,
                 discrete_space_size=50):
        """
        DeepMind Atari hyperparameters are here as an FYI

        batch_size = 32
        replay_memory_size = 1 million
        target network update freq = 10000
        replay start size = 50000 (steps of exp used to populate memory)
        learning rate = 0.00025
        total frames = 10 million
        """
        #  setup hyperparameter using DeepMind ratios as guide
        memory_length = int(total_steps * 0.1)
        self.epsilon_decay_steps = int(total_steps / 2)
        self.update_target_net = int(total_steps * 0.0125)

        #  initial number of steps to act totally random
        self.initial_random = int(total_steps * 0.1)

        #  initializing the BaseAgent class
        super().__init__(env, discount, memory_length)

        #  Q-Learning needs a discrete action space
        self.actions = self.action_space.discretize(discrete_space_size)

        #  a dictionary to setup the approximation of Q(s,a)
        #  hardcoded in for now - will eventually be pulled out
        self.model_dict = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': self.actions.shape[0],
                           'layers': [50, 25],
                           'lr': 0.0025}

        #  make our two action value functions
        self.Q_actor = Q(self.model_dict, 'actor')
        self.Q_target = Q(self.model_dict, 'target')

        #  objects to process the inputs & targets of neural networks
        self.state_processor = Standardizer(self.observation_space.shape[0])
        self.target_processor = Normalizer(1)

        #  create an object to decay epsilon
        self.e_greedy = EpsilonGreedy(decay_length=self.epsilon_decay_steps,
                                      init_random=self.initial_random)
                                      

    def _reset(self):
        """
        Resets the agent
        """
        raise NotImplementedError

    def _act(self, **kwargs):
        """
        Act using an epsilon-greedy policy

        args
            sess (tf.Session) the current tensorflow session
            obs (np.array) shape=(1, observation_space.shape[0])

        return
            action (np.array) (1, action_space.shape[0])
        """
        sess = kwargs['sess']
        observation = kwargs['obs']

        #  using our state processor to transform the observation
        observation = self.state_processor.transform(observation)

        #  get the current value of epsilon
        epsilon = self.e_greedy.epsilon
        self.memory.info['epsilon'].append(epsilon)

        if np.random.uniform() < epsilon:
            #  acting randomly
            logger.debug('epsilon {:.3f} - acting randomly'.format(epsilon))
            action = self.action_space.sample_discrete()

        else:
            #  get our Q(s,a) approximation for each action
            Q_est = self.Q_actor.predict(sess, observation)

            #  select the action with the highest Q
            action = self.actions[np.argmax(Q_est)]

            #  calculate some statistics from the Q estimates
            max_Q = np.max(Q_est)
            avg_Q = np.mean(Q_est)

            #  save some data for debugging later
            self.memory.info['max_Q_acting estimates'].append(max_Q)
            self.memory.info['avg_Q_acting estimates'].append(avg_Q)
            self.memory.info['Q act est'].extend(Q_est.flatten().tolist())

            logger.debug('using Q_actor - max(Q_est)={:.3f}'.format(max_Q))
            logger.debug('using Q_actor - avg(Q_est)={:.3f}'.format(avg_Q))

        #  make sure action is shaped correctly
        action = np.array(action).reshape(1, self.action_space.shape[0])

        return action

    def _learn(self, **kwargs):
        """
        Update Q_actor using the Bellman Equation

        observations, actions, rewards should all be either
        normalized or standardized

        Q(s',a) is calculated externally to the value function
        Q(s,a) is calculated within the value function

        args
            sess (tf.Session)
            batch (np.array): batch of experience to learn from

        returns
            train_info (dict)
        """
        sess = kwargs.pop('sess')
        batch = kwargs.pop('batch')

        obs = batch['obs']
        actions = batch['actions']
        rews = batch['rewards']
        next_obs = batch['next_obs']
        terminal = batch['terminal']

        #  process the entire batch of inputs using our state_processor
        inputs = self.state_processor.transform(obs)
        next_obs = self.state_processor.transform(next_obs)

        #  prediction of Q(s',a) for each action
        Q_next_state = self.Q_target.predict(sess, next_obs)

        #  maximize the value of the next state 
        max_Q_next_state = np.max(Q_next_state,
                                  axis=1).reshape(obs.shape[0], 1)

        #  set the max Q(s',a) equal to zero for terminal states
        max_Q_next_state[terminal] = 0

        #  use the Bellman equation with our masked max_q
        targets = rews + self.discount * max_Q_next_state

        #  save the unscaled targets so we can visualize later
        self.memory.info['unscaled_targets'].extend(list(targets.flatten()))

        #  scaling the targets by normalizing
        targets = self.target_processor.transform(targets).flatten()

        #  creating an index for the action we are training
        #  the action we are training is the action our agent chose
        #  first get a list of all possible actions
        act_list = self.actions.tolist()

        #  find the index of the action we took in the act_list
        action_index = [act_list.index(act) for act in actions.tolist()]

        #  create a 1D array of the index for each action we took
        action_index = np.array(action_index).flatten()

        assert action_index.shape[0] == actions.shape[0]
        assert targets.shape[0] == inputs.shape[0]

        #  improve our approximation of Q(s,a)
        error, loss = self.Q_actor.improve(sess, inputs, targets, action_index)

        #  save data for analysis later
        self.memory.info['train_error'].extend(error.flatten().tolist())
        self.memory.info['loss'].append(loss)

        self.memory.info['scaled_obs'].extend(inputs.flatten().tolist())
        self.memory.info['scaled_targets'].extend(targets.flatten().tolist())

        self.memory.info['max_scaled_target'].append(np.max(targets))
        self.memory.info['avg_scaled_target'].append(np.mean(targets))

        return {'error': error, 'loss': loss}

    def update_target_network(self, sess):
        """
        Copies weights from Q_actor into Q_target
        """
        logger.debug('Updating Q_target by copying weights from Q_actor')

        assert self.Q_target.scope == 'target'
        assert self.Q_actor.scope == 'actor'

        sess = self.Q_target.copy_weights(sess, parent=self.Q_actor)

        return sess
