"""v1 of the electricity flexibility environment"""

import logging

import numpy as np

from energy_py.common import DiscreteSpace, GlobalSpace
from energy_py.envs import BaseEnv


logger = logging.getLogger(__name__)


class FlexV1(BaseEnv):
    """
    Simulate electricity flexibility asset that start and stop cycles

    args
        flex_size (MW) the size of the decrease in consumption
        flex_time (int) number of 5 minute periods for both up & down cycles
        relax_time (int) number of 5 minute periods for the relax cycle
        flex_effy (float) the increase in demand (i.e. the up cycle)

    optional kwargs that are passed into BaseEnv
        dataset
        episode_length
        episode_start
        episode_random

    Unlike v0, the flex cycle in this env can be stopped

    The environment has four modes (occur in order)
        available (no change in consumption, zero reward)
        flex_down (reduction in consumption, positive reward)
        flex_up (increase in consumption, negative reward)
        relax (no change in consumption, zero reward)

    The agent can choose between three discrete actions
            0 = no op
            1 = start (if available), continue if in flex_down
            2 = stop (if in flex_down cycle)

    Once a flexibility action has started, the action continues until
    1 - the agent stops it
    2 - a maximum flex time is reached

    After the action has finished, a penalty (the flex up cycle) is paid.
    An optional length of relaxation time occurs after the flex up period

    TODO
    Currently the relaxation time is a fixed length
    - in the future this could be a function of the flex_time
    Ability to randomly start in different parts of the flex cycle
    """

    def __init__(self,
                 flex_size=2,   # MW
                 flex_time=6,   # num 5 minute periods
                 relax_time=6,  # num 5 minute periods
                 flex_effy=1.2, # percent
                 **kwargs):

        self.flex_size = float(flex_size)
        self.max_flex_time = int(flex_time)
        self.relax_time = int(relax_time)
        self.flex_effy = flex_effy

        #  counters for the different modes of operation
        self.avail = None
        self.flex_down = None
        self.flex_up = None
        self.relax = None

        #  a counter that remembers how long the flex down cycle was
        self.local_flex_time = None

        self.flex_counter = None

        #  initializing the BaseEnv class
        super().__init__(**kwargs)

        """
        SETTING THE ACTION SPACE

        Discrete action space with three choices
            0 = no op
            1 = start (if available), continue if in flex_down
            2 = stop (if in flex_down cycle)
        """
        self.action_space = GlobalSpace('action').from_spaces(
            [DiscreteSpace(3)],
            'flex_action'
        )

        obs_spaces_append = self.make_observation_append_list()
        spaces = [DiscreteSpace(1) for _ in obs_spaces_append]

        #  making a list of names for the additional observation space dims
        space_labels = ['flex_availability']
        #  +1 because the 0th element is not being in that part of the cycle
        space_labels.extend(['flex_down_{}'.format(c)
                              for c in range(self.max_flex_time + 1)])
        space_labels.extend(['flex_up_{}'.format(c)
                              for c in range(self.max_flex_time + 1)])
        space_labels.extend(['relax_{}'.format(c)
                              for c in range(self.relax_time + 1)])

        self.observation_space.extend(
            spaces, space_labels
        )

        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py flex-v1 environment>'

    def _reset(self):
        """
        Resets the environment

        returns
            observation (np.array) the initial observation
        """
        #  initialize our counters
        self.avail = 1
        self.flex_down = 0
        self.flex_up = 0
        self.relax = 0

        self.local_flex_time = 0  # num 5 min periods
        self.flex_counter = 0  # number of actions

        self.steps = 0
        self.state = self.state_space(self.steps)

        obs_append = self.make_observation_append_list()
        self.observation = self.observation_space(
            self.steps, obs_append
        )
        return self.observation

    def _step(self, action):
        """
        One step through the environment

        args
            action (np.array) shape = (1, 1)

        Action space is discrete with three choices
            0 = no op
            1 = start (if available), continue if in flex_down
            2 = stop (if in flex_down cycle)
        """
        #  if we are in the flex_down cycle, continue
        if self.flex_down > 0 and action != 2:
            self.flex_down += 1
            self.local_flex_time += 1

        #  if we are in the flex up cycle, continue
        if self.flex_up > 0:
            self.flex_up += 1

        #  if we are in the relaxation period, continue
        if self.relax > 0:
            self.relax += 1

        #  if we decide to end the flex_down cycle
        if (self.flex_down > 0 and action == 2):
            self.flex_down = 0
            self.flex_up = 1

        #  if we need to end the flex down cycle
        if self.flex_down > self.max_flex_time:
            self.flex_down = 0
            self.flex_up = 1

        #  starting the flex_down cycle
        if self.avail == 1 and action == 1:
            self.flex_down = 1
            self.avail = 0
            self.local_flex_time += 1
            self.flex_counter += 1

        #  if we need to end the flex_up cycle
        if self.flex_up > self.local_flex_time:
            self.flex_up = 0
            self.local_flex_time = 0
            self.relax = 1

        #  if we need to end the flex_up cycle
        if self.flex_up > self.max_flex_time:
            self.flex_up = 0
            self.local_flex_time = 0
            self.relax = 1

        #  if we need to end the relax
        if self.relax > self.relax_time:
            self.relax = 0
            self.avail = 1

        #  now we set the reward based on the position in the cycle
        #  default flex action of doing nothing
        flex_action = 0

        if self.flex_down > 0:
            flex_action = -self.flex_size

        if self.flex_up > 0:
            flex_action = self.flex_size * self.flex_effy

        electricity_price = self.get_state_variable(
            'C_electricity_price [$/MWh]')

        #  /12 so we get reward in terms of £/5 minutes
        reward = -flex_action * electricity_price / 12

        #  work out the flex counter
        if flex_action == 0:
            flex_counter = 'not_flexing'
        else:
            flex_counter = 'flex_action_{}'.format(self.flex_counter)

        total_counters = self.check_counters()

        if total_counters > 0:
            logger.debug('action is {}'.format(action))
            logger.debug('flex_action is {}'.format(flex_action))
            logger.debug('up {} down {} relax {} rew {}'.format(
                self.flex_up, self.flex_down, self.relax, reward))

        next_state = self.state_space(self.steps)
        obs_append = self.make_observation_append_list()
        next_observation = self.observation_space(
            self.steps, obs_append
        )
        self.steps += 1

        #  check to see if we are done
        if self.steps == (self.episode_length):
            done = True
        else:
            done = False

        info = self.update_info(steps=self.steps,
                                state=self.state,
                                observation=self.observation,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                next_observation=next_observation,
                                done=done,

                                electricity_price=electricity_price,
                                flex_down=self.flex_down,
                                flex_up=self.flex_up,
                                relax=self.relax,
                                flex_avail=self.avail,
                                flex_action=flex_action,
                                flex_counter=self.flex_counter)

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, done, info

    def make_observation_append_list(self):
        """
        Creates the list to be appended onto the observation

        returns
            append (list)

        append = [self.avail,
                  flex_down_dummies,
                  flex_up_dummies,
                  relax_dummies]

        dummies = [off, period_1, period_2 ... period_max_flex_time]
        """
        flex_down_dummies = np.zeros(self.max_flex_time + 1)
        flex_up_dummies = np.zeros(self.max_flex_time + 1)
        relax_dummies = np.zeros(self.relax_time + 1)

        flex_down_dummies[self.flex_down] = 1
        flex_up_dummies[self.flex_up] = 1
        relax_dummies[self.relax] = 1

        append = [np.array([self.avail]),
                  flex_down_dummies,
                  flex_up_dummies,
                  relax_dummies]

        return np.concatenate(append, axis=0).tolist()

    def check_counters(self):
        """
        Checks that all of the counters are set in valid positions
        """
        if self.avail != 0:
            counters = sum([self.flex_down, self.flex_up, self.relax])
            assert counters == 0

        if self.flex_down != 0:
            counters = sum([self.avail, self.flex_up, self.relax])
            assert counters == 0

        if self.flex_up != 0:
            counters = sum([self.avail, self.flex_down, self.relax])
            assert counters == 0

        if self.relax != 0:
            counters = sum([self.avail, self.flex_down, self.flex_up])
            assert counters == 0

        logger.debug('avail {} flex_down {} flex_up {} relax {}' \
                     .format(self.avail,
                             self.flex_down,
                             self.flex_up,
                             self.relax))

        return sum([self.relax, self.flex_down, self.flex_up])
