import logging

from energy_py import DiscreteSpace, GlobalSpace
from energy_py.envs import TimeSeriesEnv

logger = logging.getLogger(__name__)


class FlexEnv(TimeSeriesEnv):
    def __init__(self,
                 data_path,
                 log_path=None,
                 episode_length=48,
                 episode_start=0,
                 episode_random=False,
                 flex_initial_size=1,  # in MW
                 flex_final_size=-1,
                 flex_time=6,  # num 5 minute periods
                 relax_time=12):  # num 5 min periods

        #  technical energy inputs
        self.flex_initial_size = float(flex_initial_size)
        self.flex_final_size = float(flex_final_size)

        self.flex_time = int(flex_time)
        self.relax_time = int(relax_time)

        if self.flex_initial_size + self.flex_final_size != 0:
            raise ValueError('Your flex actions are not equal and opposite')

        self.electricity_price = None
        self.flex_initial = None
        self.flex_final = None
        self.relax = None
        self.flex_avail = None
        self.flex_action = None

        scalars = {'electricity_price': self.electricity_price,
                   'flex_up': self.flex_final,
                   'flex_down': self.flex_initial,
                   'relax': self.relax,
                   'flex_avail': self.flex_avail,
                   'flex_action': self.flex_action}

        #  init the parent TimeSeriesEnv
        super().__init__(data_path,
                         episode_length,
                         episode_start,
                         episode_random,
                         log_path=log_path,
                         tb_scalars=scalars)

    def __repr__(self): return '<energy_py flexibility environment>'

    def _reset(self):
        """
        Resets the environment

        returns
            observation (np.array) the initial observation

        SETTING THE ACTION SPACE

        Single action - whether to start the flex asset or not
            0 = do nothing
            1 = start flex cycle

        Once flex cycle is started it runs for the flex_time
        After flex_time is over, relax_time starts
        """
        self.action_space = GlobalSpace([DiscreteSpace(1)])

        #  initialize all of the
        self.flex_avail = 1  # 0=not available, 1=available
        self.flex_initial = 0
        self.flex_final = 0
        self.relax = 0

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

        """
        Resetting steps, state, observation, done status
        """
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
        action = action[0][0]
        assert action >= self.action_space.spaces[0].low
        assert action <= self.action_space.spaces[0].high

        #  probably a bit excessive to check twice (I check again below)
        total_counters = self.check_counters()

        #  if we are in the initial flex cycle, continue that
        if self.flex_initial > 0:
            self.flex_initial += 1

        #  if we are in the flex up cycle, continue
        if self.flex_final > 0:
            self.flex_final += 1

        #  if we are in the relaxation period, continue that
        if self.relax > 0:
            self.relax += 1

        #  if we are ending the initial flex cycle, and starting flex up
        if self.flex_initial > self.flex_time:
            self.flex_initial = 0
            self.flex_final = 1

        #  if we are ending the flex up cycle, and starting relaxation
        if self.flex_final > self.flex_time:
            self.flex_final = 0
            self.relax = 1

        #  ending the relaxation period
        if self.relax > self.relax_time:
            self.relax = 0
            self.flex_avail = 1

        #  if we are not doing anything but want to start the flex cycle
        total_counters = sum([self.flex_final, self.flex_initial, self.relax])
        if total_counters == 0 and action == 1:
            self.flex_initial = 1
            self.flex_avail = 0

        #  we set the default action to do nothing
        flex_action = 0

        #  we set the flex action to do something if we are flexing

        if self.flex_initial > 0:
            flex_action = self.flex_initial_size

        if self.flex_final > 0:
            flex_action = self.flex_final_size

        #  now we set reward based on whether we are in a cycle or not
        #  /12 so we get reward in terms of £/5 minutes
        reward = flex_action * self.electricity_price / 12

        total_counters = self.check_counters()

        logger.debug('step {} elect. price {}'.format(
            self.observation_ts.index[self.steps], self.electricity_price))

        if total_counters > 0:
            logger.debug('action is {}'.format(action))
            logger.debug('flex_action is {}'.format(flex_action))
            logger.debug('up {} down {} relax {} rew {}'.format(
                self.flex_final, self.flex_initial, self.relax, reward))

        self.tb_helper.write_summaries()

        self.steps += 1
        next_state = self.get_state(self.steps, append=self.flex_avail)
        next_observation = self.get_observation(self.steps, self.flex_avail)

        #  check to see if we are done
        if self.steps == (self.episode_length - 1):
            self.done = True

        #  moving to the next time step
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def check_counters(self):
        """
        Helper function to check that the counters are OK
        """

        if self.flex_final != 0:
            assert self.flex_initial == 0
            assert self.relax == 0
            assert self.flex_avail == 0

        if self.flex_initial != 0:
            assert self.flex_final == 0
            assert self.relax == 0
            assert self.flex_avail == 0

        if self.relax != 0:
            assert self.flex_initial == 0
            assert self.flex_final == 0
            assert self.flex_avail == 0

        total_counters = sum([self.flex_final, self.flex_initial, self.relax])
        if self.flex_avail == 0:
            assert total_counters != 0

        if self.flex_avail == 1:
            assert total_counters == 0

        return total_counters
