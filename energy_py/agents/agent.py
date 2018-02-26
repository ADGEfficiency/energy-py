import logging

import tensorflow as tf

from energy_py.agents import memories
from energy_py import processors

logger = logging.getLogger(__name__)


class BaseAgent(object):
    """
    The energy_py base agent class

    The main methods of this class are
        reset
        act
        learn

    All agents should override the following methods
        _reset
        _act
        _learn
        _output_results

    args
        env (object) energy_py environment
        discount (float) discount rate aka gamma
        memory_length (int) number of experiences to store
    """

    def __init__(self,
                 env,
                 discount,
                 memory_length,
                 memory_type='deque',
                 observation_processor=None,
                 action_processor=None,
                 target_processor=None,
                 act_path=None,
                 learn_path=None,
                 **kwargs):

        self.env = env
        self.discount = discount

        self.observation_space = env.observation_space
        self.obs_shape = env.observation_space.shape

        self.action_space = env.action_space
        self.action_shape = env.action_space.shape

        self.memory = memories[memory_type](memory_length,
                                            self.obs_shape,
                                            self.action_shape)

        #  a counter our agent can use as it sees fit
        self.counter = 0

        #  inital number of steps not to learn from
        #  defaults at 0
        self.initial_random = 0

        #  optional objects to process arrays before they hit neural networks
        if observation_processor:
            self.observation_processor = processors[observation_processor]

        if action_processor:
            self.action_processor = processors[action_processor]

        if target_processor:
            self.target_processor = processors[target_processor]

        #  optional tensorflow FileWriters for acting and learning
        if act_path:
            self.acting_writer = tf.summary.FileWriter(act_path)

        if learn_path:
            self.learning_writer = tf.summary.FileWriter(learn_path,
                                                         graph=self.sess.graph)

    def _reset(self): raise NotImplementedError

    def _act(self, observation): raise NotImplementedError

    def _learn(self, **kwargs): raise NotImplementedError

    def reset(self):
        """
        Resets the agent internals.
        """
        self.memory.reset()
        return self._reset()

    def act(self, observation):
        """
        Action selection by agent.

        args
            observation (np array) shape=(1, observation_dim)

        return
            action (np array) shape=(1, num_actions)
        """
        logger.debug('Agent is acting')

        if hasattr(self, 'observation_processor'):
            observation = self.observation_processor.transform(observation)

        #  some environments (i.e. gym) return observations as flat arrays
        #  energy_py agents use arrays of shape(batch_size, *shape)
        if observation.ndim == 1:
            observation = observation.reshape(1, *self.obs_shape)

        assert observation.shape[0] == 1
        return self._act(observation)

    def learn(self, **kwargs):
        """
        Agent learns from experience.

        args
            batch (dict) batch of experience
            sess (tf.Session)

        return
            training_history (object) info about learning (i.e. loss)
        """
        logger.debug('Agent is learning')
        return self._learn(**kwargs)

    def remember(self, observation, action, reward, next_observation, done):
        """
        Store experience in the agent's memory.

        args
            observation (np.array)
            action (np.array)
            reward (np.array)
            next_observation (np.array)
            done (np.array)
        """
        observation = observation.reshape(-1, *self.obs_shape)
        next_observation = next_observation.reshape(-1, *self.obs_shape)

        if hasattr(self, 'observation_processor'):
            observation = self.observation_processor.transform(observation)
            next_observation = self.observation_processor.transform(next_observation)

        if hasattr(self, 'action_processor'):
            action = self.action_processor.transform(action)

        return self.memory.remember(observation, action, reward,
                                    next_observation, done)


