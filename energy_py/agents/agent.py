import logging
import os

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

    Some agents will also override
        _load_brain
        _save_brain
        _output_results

    args
        env      : energy_py environment
        discount : float : discount rate (gamma)

    methods
        all_state_actions : used to create all combinations of state across the
                            action space
    """

    def __init__(self,
                 env,
                 discount,
                 brain_path=[],
                 memory_length=100000):

        self.env = env
        self.discount = discount
        self.brain_path = brain_path

        #  use the env to setup the agent
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_info = self.env.observation_info

        #  create a memory for the agent
        #  object to hold all of the agents experience
        #  TODO does memory need all this now???
        self.memory = Memory(self.observation_space,
                             self.action_space,
                             self.discount,
                             memory_length=memory_length)

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError

    def _act(self, **kwargs): raise NotImplementedError

    def _learn(self, **kwargs): raise NotImplementedError

    def _load_brain(self): raise NotImplementedError

    def _save_brain(self): raise NotImplementedError

    def _output_results(self): raise NotImplementedError

    def reset(self):
        """
        Resets the agent
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        return self._reset()

    def act(self, **kwargs):
        """
        Action selection by agent

        args
            observation (np array) : shape=(1, observation_dim)

        return
            action (np array) : shape=(1, num_actions)
        """
        logger.debug('Agent is acting')
        return self._act(**kwargs)

    def learn(self, **kwargs):
        """
        Agent learns from experience

        Use **kwargs for flexibility

        return
            training_history (object) : info about learning (i.e. loss)
        """
        logger.debug('Agent is learning')
        return self._learn(**kwargs)

    def load_brain(self):
        """
        Agent can load previously created memories, policies or value functions
        """
        logger.info('Loading agent brain')
        memory_path = os.path.join(self.brain_path, 'memory.pickle')
        self.memory = self.load_pickle(memory_path)

        return self._load_brain()

    def save_brain(self):
        """
        Agent can save previously created memories, policies or value functions
        """
        logger.info('Saving agent brain')

        #  we save the agent memory
        memory_path = os.path.join(self.brain_path, 'memory.pickle')
        self.dump_pickle(self.memory, memory_path)

        return self._save_brain()

    def output_results(self):
        """
        Agent can load previously created memories, policies or value functions
        """
        return self.memory.output_results()


class EpsilonGreedy(object):
    """
    A class to perform epsilon greedy action selection.

    Decay is done linearly.

    Decay occurs every time we call get_epsilon.

    TODO update with logger
    """
    def __init__(self,
                 random_start,
                 decay_steps,
                 epsilon_start=1.0,
                 epsilon_end=0.1):

        #  we calculate a linear coefficient to decay with
        self.linear_coeff = (epsilon_end - epsilon_start) / decay_steps

        self.random_start = random_start
        self.decay_steps = decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

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

        else:
            self._epsilon = self.epsilon_end

        self.steps += 1
        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
