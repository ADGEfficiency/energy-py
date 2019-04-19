from collections import defaultdict

import tensorflow as tf

import energypy


class BaseAgent(object):
    """ The energypy base agent class """

    def __init__(
            self,
            env,
            sess=None,
            total_steps=0,

            memory_type='deque',
            memory_length=10000,
            load_memory_path=None,

            min_reward=-10,
            max_reward=10,
            tensorboard_dir=None,
            **kwargs
    ):

        self.sess = sess
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.memory = energypy.make_memory(
            memory_id=memory_type,
            env=env,
            size=memory_length,
            load_path=load_memory_path
        )

        #  reward clipping
        self.min_reward = min_reward
        self.max_reward = max_reward

        self.act_step = 0
        self.learn_step = 0

        self.summaries = {
            'acting': [],
            'learning': []
        }

        if tensorboard_dir:
            self.writers = {
                'acting': tf.summary.FileWriter(tensorboard_dir+'/acting'),
                'learning': tf.summary.FileWriter(tensorboard_dir+'/learning')
            }

        #  TODO
        self.filters = None
        self.kernels = None
        self.strides = None

    def reset(self):
        """
        Resets the agent internals
        """
        self.memory.reset()
        self.act_step = 0
        self.learn_step = 0

        return self._reset()

    def act(self, observation):
        """
        Action selection by agent

        args
            observation (np array) shape=(1, observation_dim)

        return
            action (np array) shape=(1, num_actions)
        """
        self.act_step += 1

        return self._act(observation.reshape(1, *self.observation_space.shape))

    def learn(self, **kwargs):
        """
        Agent learns from experience

        args
            batch (dict) batch of experience
            sess (tf.Session)

        return
            training_history (object) info about learning (i.e. loss)
        """
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
        if self.min_reward and self.max_reward:
            reward = max(self.min_reward, min(reward, self.max_reward))

        return self.memory.remember(
            observation, action, reward, next_observation, done
        )
