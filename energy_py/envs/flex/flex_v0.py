"""v0 of the flexibility environment"""

import logging

from energy_py.envs import BaseEnv
from energy_py.common import DiscreteSpace, GlobalSpace


logger = logging.getLogger(__name__)


class FlexV0(BaseEnv):
    """
    Simulate electricity flexibility asset that can only start full cycles 

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

    Model simulates a flexibility cycle of three stages
    - flex up = increased consumption
    - flex down = decreased consumption
    - relaxation = agent has to wait until starting the next cycle

    The agent can choose between three discrete actions
        0 = no_op
        1 = start flex down -> flex up cycle
        2 = start flex up -> flex down cycle

    Once the cycle has started it runs until completion
    """
    def __init__(self,
                 flex_size=2,   # MW
                 flex_time=6,   # num 5 minute periods
                 relax_time=6,  # num 5 minute periods
                 flex_effy=1.2, # percent
                 **kwargs):

        #  technical energy inputs
        self.flex_down_size = -float(flex_size)
        #  flex up we increase consumption by more than when we flex down
        self.flex_up_size = float(flex_size) * float(flex_effy)

        #  assume that flex down & up times are the same
        #  model is built to change this eaisly
        self.flex_down_time = int(flex_time)
        self.flex_up_time = int(flex_time)
        self.relax_time = int(relax_time)

        self.flex_down = None
        self.flex_up = None
        self.relax = None
        self.avail = None

        #  local action remembers which action was taken during a cycle
        #  set when a cycle starts, reset inbetween cycles
        self.local_action = None

        #  counts the number of flexes during an episode
        self.flex_counter = 0

        #  initialize the BaseEnv - should be done before setting the
        #  action and observation spaces
        super().__init__(**kwargs)

        """
        SETTING THE ACTION SPACE

        Single action - whether to start the flex asset or not
            0 = do nothing
            1 = start flex down then flex up cycle
            2 = start flex up then flex down cycle

        Once flex cycle is started it runs for the flex_time
        After flex_time is over, relax_time starts
        """
        #  create the action space object
        self.action_space = GlobalSpace([DiscreteSpace(3)])

        """
        SETTING THE OBSERVATION SPACE
        """
        #  add infomation onto our observation
        spaces = [DiscreteSpace(1),
                  DiscreteSpace(self.flex_down_time),
                  DiscreteSpace(self.flex_up_time),
                  DiscreteSpace(self.relax_time)]

        #  names of these additional observation variables
        space_labels = ['flex_availability',
                        'flex_down_cycle',
                        'flex_up_cycle',
                        'relax_cycle']

        self.observation_space = self.make_observation_space(spaces,
                                                             space_labels)

        #  set the initial observation by resetting the environment
        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py flex-v0 environment>'

    def _reset(self):
        """
        Resets the environment

        returns
           observation (np.array) the initial observation
        """
        #  initialize all of the flex counters
        self.avail = 1  # 0=not available, 1=available
        self.flex_down = 0
        self.flex_up = 0
        self.relax = 0
        self.flex_counter = 0

        #  initialize our local action being the no_op action
        self.local_action = 0

        #  Resetting steps, state, observation
        self.steps = 0
        self.state = self.get_state(steps=self.steps)
        self.observation = self.get_observation(self.steps,
                                                append=[self.avail,
                                                        self.flex_down,
                                                        self.flex_up,
                                                        self.relax])
        return self.observation

    def _step(self, action):
        """
        One step through the environment

        args
            action (np.array) shape=(1, 1)

        returns
            observation (np.array) shape=(1, self.observation_space.shape*)
            reward (float)
            done (bool)
            info (dict)

        Flex asset is dispatched if action=1 and not already in a flex cycle
        or relaxing.
        """
        #  pull the electricity price out of the state
        price_index = self.state_info.index('C_electricity_price_[$/MWh]')
        electricity_price = self.state[0][price_index]

        assert action.shape == (1, 1)

        #  grab the action
        action = action[0][0]
        assert action >= self.action_space.spaces[0].low
        assert action <= self.action_space.spaces[0].high

        #  probably a bit excessive to check counters twice
        #  (I check again below)
        total_counters = self.check_counters()

        #  if we are in the initial flex cycle, continue that
        if self.flex_down > 0:
            self.flex_down += 1

        #  if we are in the flex up cycle, continue
        if self.flex_up > 0:
            self.flex_up += 1

        #  if we are in the relaxation period, continue that
        if self.relax > 0:
            self.relax += 1

        #  if we are ending a flex down cycle
        if self.flex_down > self.flex_down_time:
            self.flex_down = 0
            #  check if we need to start the flex up cycle now
            if self.local_action == 1:
                self.flex_up = 1
                self.local_action = 0
            #  otherwise we start the relaxation period
            else:
                self.relax = 1

        #  if we are ending a flex up cycle
        if self.flex_up > self.flex_up_time:
            self.flex_up = 0
            #  check if we need to start flex down cycle now
            if self.local_action == 2:
                self.flex_down = 1
                self.local_action = 0
            #  otherwise we start the relaxation period
            else:
                self.relax = 1

        #  ending the relaxation period
        if self.relax > self.relax_time:
            self.relax = 0
            self.avail = 1

        #  if we are not doing anything but want to start the flex down cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 1:
            self.flex_down = 1
            self.avail = 0
            self.local_action = action
            self.flex_counter += 1

        #  if we are not doing anything but want to start the flex up cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 2:
            self.flex_up = 1
            self.avail = 0
            self.local_action = action
            self.flex_counter += 1

        #  we set the default action to do nothing (in MW)
        flex_action = 0

        #  we set the flex action to do something if we are flexing
        if self.flex_down > 0:
            flex_action = self.flex_down_size

        if self.flex_up > 0:
            flex_action = self.flex_up_size

        #  now we set reward based on the flex_action
        #  /12 so we get reward in terms of $/5 minutes
        reward = - flex_action * electricity_price / 12

        #  another paranoid check of the counters
        total_counters = self.check_counters()

        #  log the asset flex status
        logger.debug('flex_action_num_{}_{}_MW'.format(self.flex_counter,
                                                       flex_action))
        #  log the step and electricity price
        logger.debug('step {} elect. price {} $/MWh'.format(
            self.observation_ts.index[self.steps], electricity_price))

        if total_counters > 0:
            logger.debug('action is {}'.format(action))
            logger.debug('flex_action is {}'.format(flex_action))
            logger.debug('up {} down {} relax {} rew {}'.format(
                self.flex_up, self.flex_down, self.relax, reward))

        next_state = self.get_state(self.steps)
        next_observation = self.get_observation(self.steps,
                                                append=[self.avail,
                                                        self.flex_down,
                                                        self.flex_up,
                                                        self.relax])
        self.steps += 1

        #  check to see if we are done
        done = False
        if self.steps == (self.episode_length):
            done = True

        self.info = self.update_info(steps=self.steps,
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

        #  transition to the next step in the environment
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, done, self.info

    def check_counters(self):
        """
        Helper function to check that the counters are in viable positions

        returns
            total_counters (int) sum of all the counters
        """
        total_counters = sum([self.flex_up, self.flex_down, self.relax])

        if self.avail == 0:
            assert total_counters != 0

        if self.avail == 1:
            assert total_counters == 0

        if self.flex_up != 0:
            assert self.flex_down == 0
            assert self.relax == 0
            assert self.avail == 0

        if self.flex_down != 0:
            assert self.flex_up == 0
            assert self.relax == 0
            assert self.avail == 0

        if self.relax != 0:
            assert self.flex_down == 0
            assert self.flex_up == 0
            assert self.avail == 0

        return total_counters
