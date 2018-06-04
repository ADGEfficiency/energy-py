import logging

import tensorflow as tf

from energy_py.common.memories import memory_register

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
    Optionally override
        _learn

    """

    def __init__(self,
                 sess,
                 env,
                 memory_type='priority',
                 memory_length=10000,
                 min_reward=-10,
                 max_reward=10,
                 observation_processor=None,
                 action_processor=None,
                 target_processor=None,
                 act_path=None,
                 learn_path=None,
                 **kwargs):

        self.sess = sess
        self.env = env

        self.observation_space = env.observation_space
        self.obs_shape = env.observation_space.shape
        self.observation_info = env.observation_info

        self.action_space = env.action_space
        self.action_shape = env.action_space_shape

        #  sending total_steps and initial_random into the memory init
        self.memory_type = memory_type

        self.memory = memory_register[memory_type](
            memory_length,
            self.obs_shape,
            self.action_shape
        )

        #  reward clipping
        self.min_reward = min_reward
        self.max_reward = max_reward

        #  keep two step counters internally in the agent
        self.act_step = 0
        self.learn_step = 0

        #  optional objects to process arrays before they hit neural networks
        if observation_processor:
            self.observation_processor = processors[observation_processor]()

        if action_processor:
            self.action_processor = processors[action_processor]()

        if target_processor:
            self.target_processor = processors[target_processor]()

        #  optional tensorflow FileWriters for acting and learning
        if act_path and hasattr(self, 'sess'):
            self.acting_writer = tf.summary.FileWriter(act_path)

        if learn_path and hasattr(self, 'sess'):
            self.learning_writer = tf.summary.FileWriter(learn_path)

    def _reset(self): raise NotImplementedError

    def _act(self, observation): raise NotImplementedError

    def _learn(self, **kwargs): pass

    def reset(self):
        """
        Resets the agent internals
        """
        logger.debug('Resetting the agent internals')

        self.memory.reset()

        return self._reset()

    def act(self, observation):
        """
        Action selection by agent

        args
            observation (np array) shape=(1, observation_dim)

        return
            action (np array) shape=(1, num_actions)
        """
        logger.debug('Agent is acting')

        #  some environments (i.e. gym) return observations as flat arrays
        #  energy_py agents use arrays of shape(batch_size, *shape)
        observation = observation.reshape(1, *self.obs_shape)

        if hasattr(self, 'observation_processor'):
            observation = self.observation_processor.transform(observation)

        self.act_step += 1

        return self._act(observation)

    def learn(self, **kwargs):
        """
        Agent learns from experience

        args
            batch (dict) batch of experience
            sess (tf.Session)

        return
            training_history (object) info about learning (i.e. loss)
        """
        logger.debug('Agent is learning')
        self.learn_step += 1

        return self._learn(**kwargs)

    def remember(self, observation, action, reward, next_observation, done):
        """
        Store experience in the agent's memory

        args
            observation (np.array)
            action (np.array)
            reward (np.array)
            next_observation (np.array)
            done (np.array)
        """
        logger.debug('Agent is remembering')

        observation = observation.reshape(-1, *self.obs_shape)
        next_observation = next_observation.reshape(-1, *self.obs_shape)

        if hasattr(self, 'observation_processor'):
            observation = self.observation_processor.transform(observation)
            next_observation = self.observation_processor.transform(next_observation)

        if hasattr(self, 'action_processor'):
            action = self.action_processor.transform(action)

        #  reward clipping
        if self.min_reward and self.max_reward:
            reward = max(self.min_reward, min(reward, self.max_reward))

        return self.memory.remember(observation, action, reward,
                                    next_observation, done)
