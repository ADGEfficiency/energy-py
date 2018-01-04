import collections
import logging

import pandas as pd


logger = logging.getLogger(__name__)


class BaseEnv(object):
    """
    The parent class for energy_py environments
    inspired by the gym.Env class

    The methods of this class are:
        step
        reset

    To implement an environment:
      - inherit from this class

      - override the following methods in your child:
        _step()
        _reset()
        _output_results()

      - set the following attributes:
        action_space
        observation_space

    """

    def __init__(self):
        self.observation = self.reset(episode='none')

    # Override in subclasses
    def _step(self, action): raise NotImplementedError

    def _reset(self): raise NotImplementedError

    #  Set these in ALL subclasses
    action_space = None
    observation_space = None

    def reset(self, episode):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (np array): the initial observation
        """
        self.episode = episode
        logger.debug('Reset environment')

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        return self._reset()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        User is responsible for resetting after episode end.

        The step function should progress in the following order:
        - action = a[1]
        - reward = r[1]
        - next_state = next_state[1]
        - update_info()
        - step += 1
        - self.state = next_state[1]

        step() returns the observation - not the state!

        args
            action (object) an action provided by the environment
            episode (int) the current episode number

        returns:
            observation (np array) agent's observation of the environment
            reward (np.float)
            done (boolean)
            info (dict) auxiliary information
        """
        logger.debug('Episode {} - Step {}'.format(self.episode, self.steps))

        assert action.shape == (1, self.action_space.shape[0])

        return self._step(action)

    def output_results(self):
        """
        Adds the .info dictionary to the outputs dictionary.
        Makes a dataframe from self.info.

        returns
            self._output_results() function set in child class
        """
        logger.debug('Outputting resuts')
        #  add the self.info dictionary into our outputs dictionary
        self.outputs['info'] = self.info

        #  make a dataframe from self.info
        self.outputs['df_env_info'] = pd.DataFrame.from_dict(self.info)

        return self._output_results()

    def update_info(self, **kwargs):
        """
        Helper function to update the self.info dictionary.
        """
        for name, data in kwargs.items():
            self.info[name].append(data)

        return self.info
