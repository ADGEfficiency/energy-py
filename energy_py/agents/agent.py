import logging

from energy_py.agents.memory import Memory

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

    Two counters are used
        self.count is the total number of steps the object has seen
        self.decay_count is the number of steps in the decary period

    args
        decay_length (int) len of the linear decay period
        init_random (int) num steps to act fully randomly at start
        eps_start (float) initial value of epsilon
        eps_end (float) final value of epsilon
    """

    def __init__(self,
                 decay_length,
                 init_random=0,
                 eps_start=1.0,
                 eps_end=0.1):

        self.decay_length = int(decay_length)
        self.init_random = int(init_random)
        self.min_start = self.init_random + self.decay_length

        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)

        eps_delta = self.eps_start - self.eps_end
        self.coeff = - eps_delta / self.decay_length

        self.reset()

    def reset(self):
        self.count = 0
        self.decay_count = 0

    @property
    def epsilon(self):
        #  move the counter each step
        self.count += 1

        if self.count <= self.init_random:
            self._epsilon = 1.0

        if self.count > self.init_random and self.count <= self.min_start:
            self._epsilon = self.coeff * self.decay_count + self.eps_start
            self.decay_count += 1

        if self.count > self.min_start:
            self._epsilon = self.eps_end

        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
