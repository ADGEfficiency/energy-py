import logging

from energy_py import DiscreteSpace, GlobalSpace
from energy_py.envs import BaseEnv

logger = logging.getLogger(__name__)


class Flex(BaseEnv):
    """
    An environment to simulate electricity flexibility responding to the price
    of electricity.

    Model simulates a flexibiltiy cycle of three stages
    - flex up = increased consumption
    - flex down = decreased consumption
    - relaxation = agent has to wait until starting the next cycle

    Action space is discrete:
        0 = do nothing
        1 = start flex down then flex up cycle
        2 = start flex up then flex down cycle
    """
    def __init__(self,
                 data_path,
                 episode_length=48,
                 episode_start=0,
                 episode_random=False,
                 flex_size=2,  # MW
                 flex_time=6,  # 5 minute periods
                 relax_time=12,  # 5 minute periods
                 flex_effy=1.2):  # additional consumption in flex up

        #  technical energy inputs
        self.flex_down_size = float(flex_size)
        #  flex up we increase consumption by more than when we flex down
        self.flex_up_size = -float(flex_size * flex_effy)

        #  assume that flex down & up times are the same
        #  model is built to change this eaisly
        self.flex_down_time = int(flex_time)
        self.flex_up_time = int(flex_time)
        self.relax_time = int(relax_time)

        self.electricity_price = None
        self.flex_down = None
        self.flex_up = None
        self.relax = None
        self.flex_avail = None
        self.flex_action = None
        self.action = None
        self.flex_counter = 0
        self.action_counter = 0

        super().__init__(data_path,
                         episode_length,
                         episode_start,
                         episode_random)
        """
        SETTING THE ACTION SPACE

        Single action - whether to start the flex asset or not
            0 = do nothing
            1 = start flex down then flex up cycle
            2 = start flex up then flex down cycle

        Once flex cycle is started it runs for the flex_time
        After flex_time is over, relax_time starts
        """
        self.action_space = GlobalSpace([DiscreteSpace(2)])

        """
        SETTING THE OBSERVATION SPACE

        Set in the parent class TimeSeriesEnv
        Append the flex asset availability to send to the agent
        TODO probably worth appending some more stuff as well!
        """
        obs_spc, self.observation_ts, self.state_ts = self.get_state_obs()

        #  add on a space to represent the flex availability
        obs_spc.append(DiscreteSpace(1))
        self.observation_info.append('flex_availability')
        self.observation_space = GlobalSpace(obs_spc)

        #  set the initial observation by resetting the environment
        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py FLEX environment>'

    def _reset(self):
        """
        Resets the environment.

        returns
            observation (np.array) the initial observation
        """
        #  initialize all of the flex counters
        self.flex_avail = 1  # 0=not available, 1=available
        self.flex_down = 0
        self.flex_up = 0
        self.relax = 0

        #  initialize our action checker
        self.action = 0
        #  our env also keeps a list of the times when we started flexing
        self.flex_start_steps = []

        #  Resetting steps, state, observation, done status
        self.steps = 0
        self.state = self.get_state(steps=self.steps, append=self.flex_avail)
        self.observation = self.get_observation(self.steps, self.flex_avail)
        self.done = False

        return self.observation

    def _step(self, action):
        """
        One step through the environment.

        Flex asset is dispatched if action=1 and not already in a flex cycle
        or relaxing.

        args
            action (np.array) shape=(1, 1)

        returns
            observation (np.array) shape=(1, self.observation_space.shape[0])
            reward (float)
            done (bool)
            info (dict)
        """
        #  pull the electricity price out of the state
        elect_price_index = self.state_info.index('C_electricity_price_[$/MWh]')
        self.electricity_price = self.state[0][elect_price_index]

        #  grab the action
        assert action.shape == (1, 1)

        action = action[0][0]
        assert action >= self.action_space.spaces[0].low
        assert action <= self.action_space.spaces[0].high

        #  probably a bit excessive to check twice (I check again below)
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
            if self.action == 1:
                self.flex_up = 1
                self.action = 0
            #  otherwise we start the relaxation period
            else:
                self.relax = 1

        #  if we are ending a flex up cycle
        if self.flex_up > self.flex_up_time:
            self.flex_up = 0
            #  check if we need to start flex down cycle now
            if self.action == 2:
                self.flex_down = 1
                self.action = 0
            #  otherwise we start the relaxation period
            else:
                self.relax = 1

        #  ending the relaxation period
        if self.relax > self.relax_time:
            self.relax = 0
            self.flex_avail = 1

        #  if we are not doing anything but want to start the flex down cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 1:
            self.flex_down = 1
            self.flex_avail = 0
            self.action = action
            self.flex_counter += 1

        #  if we are not doing anything but want to start the flex up cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 2:
            self.flex_up = 1
            self.flex_avail = 0
            self.action = action
            self.flex_counter += 1

        #  we set the default action to do nothing
        flex_action = 0

        #  we set the flex action to do something if we are flexing
        if self.flex_down > 0:
            flex_action = self.flex_down_size

        if self.flex_up > 0:
            flex_action = self.flex_up_size

        #  now we set reward based on whether we are in a cycle or not
        #  /12 so we get reward in terms of £/5 minutes
        reward = flex_action * self.electricity_price / 12

        if flex_action == 0:
            flex_counter = 'not_flexing'
        else:
            flex_counter = 'flex_action_{}'.format(self.flex_counter)

        total_counters = self.check_counters()

        logger.debug('step {} elect. price {}'.format(
            self.observation_ts.index[self.steps], self.electricity_price))

        if total_counters > 0:
            logger.debug('action is {}'.format(action))
            logger.debug('flex_action is {}'.format(flex_action))
            logger.debug('up {} down {} relax {} rew {}'.format(
                self.flex_up, self.flex_down, self.relax, reward))

        self.steps += 1
        next_state = self.get_state(self.steps, append=self.flex_avail)
        next_observation = self.get_observation(self.steps, self.flex_avail)

        #  check to see if we are done
        if self.steps == (self.episode_length - 1):
            self.done = True

        self.info = self.update_info(steps=self.steps,
                                     state=self.state,
                                     observation=self.observation,
                                     action=action,
                                     reward=reward,
                                     next_state=next_state,
                                     next_observation=next_observation,
                                     done=self.done,

                                     electricity_price=self.electricity_price,
                                     flex_down=self.flex_down,
                                     flex_up=self.flex_up,
                                     relax=self.relax,
                                     flex_avail=self.flex_avail,
                                     flex_action=flex_action,
                                     flex_counter=flex_counter)

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def check_counters(self):
        """
        Helper function to check that the counters are OK
        """

        if self.flex_up != 0:
            assert self.flex_down == 0
            assert self.relax == 0
            assert self.flex_avail == 0

        if self.flex_down != 0:
            assert self.flex_up == 0
            assert self.relax == 0
            assert self.flex_avail == 0

        if self.relax != 0:
            assert self.flex_down == 0
            assert self.flex_up == 0
            assert self.flex_avail == 0

        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if self.flex_avail == 0:
            assert total_counters != 0

        if self.flex_avail == 1:
            assert total_counters == 0

        return total_counters
