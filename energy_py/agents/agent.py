import logging

from energy_py.agents.memory import Memory
from energy_py import Utils

logger = logging.getLogger(__name__)


class BaseAgent(Utils):
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
                 memory_length=100000):

        self.env = env
        self.discount = discount

        #  use the env to setup the agent
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        #  create a memory for the agent
        self.memory = Memory(self.observation_space,
                             self.action_space,
                             self.discount,
                             memory_length=memory_length)

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError

    def _act(self, **kwargs): raise NotImplementedError

    def _learn(self, **kwargs): raise NotImplementedError

    def _output_results(self): raise NotImplementedError

    def reset(self):
        """
        Resets the agent.
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        return self._reset()

    def act(self, **kwargs):
        """
        Action selection by agent.

        args
            observation (np array) shape=(1, observation_dim)
            sess (tf.Session)

        return
            action (np array) shape=(1, num_actions)
        """
        logger.debug('Agent is acting')
        return self._act(**kwargs)

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

    def output_results(self):
        """
        Calls the memory output_results method.
        """
        return self.memory.output_results()


class EpsilonGreedy(object):
    """
    A class to decay epsilon.  Epsilon is used in e-greedy action selection.

    Initially act totally random, then linear decay to a minimum.

    args
        random_start (int) number of steps to act totally random
        decay_steps (int) number of steps to decay epsilon from start to end
        epsilon_start (float)
        epsilon_end (float)
    """
    def __init__(self,
                 random_start,
                 decay_steps,
                 epsilon_start=1.0,
                 epsilon_end=0.1):
        self.random_start = int(random_start)
        self.decay_steps = int(decay_steps)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)

        #  we calculate a linear coefficient to decay with
        self.linear_coeff = (epsilon_end - epsilon_start) / decay_steps

        self.reset()

    def reset(self):
        self.steps = 0
        self._epsilon = self.epsilon_start

    @property
    def epsilon(self):
        if self.steps < self.random_start:
            self._epsilon = 1

        elif self.steps >= self.random_start and self.steps < self.decay_steps:
            self._epsilon = self.linear_coeff * self.steps + self.epsilon_start
            self.steps += 1

        else:
            self._epsilon = self.epsilon_end

        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
