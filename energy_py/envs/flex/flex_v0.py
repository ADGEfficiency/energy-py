import logging

from energy_py import DiscreteSpace, GlobalSpace
from energy_py.envs import BaseEnv

logger = logging.getLogger(__name__)


class FlexV0(BaseEnv):
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

    Attributes
        local_action (int) remembers which action was taken during a cycle

    kwargs that can be passed to the parent class BaseEnv
        dataset_name
        episode_length
        episode_start
        episode_random
    """
    def __init__(self,
                 flex_size=2,  # MW
                 flex_time=6,  # 5 minute periods
                 relax_time=12,  # 5 minute periods
                 flex_effy=1.2,
                 **kwargs):  # additional consumption in flex up

        #  technical energy inputs
        self.flex_down_size = float(flex_size)
        #  flex up we increase consumption by more than when we flex down
        self.flex_up_size = -float(flex_size * flex_effy)

        #  assume that flex down & up times are the same
        #  model is built to change this eaisly
        self.flex_down_time = int(flex_time)
        self.flex_up_time = int(flex_time)
        self.relax_time = int(relax_time)

        self.flex_down = None
        self.flex_up = None
        self.relax = None
        self.flex_avail = None
        self.flex_action = None

        #  local action remembers which action was taken during a cycle
        #  reset inbetween cycles
        self.local_action = None

        #  counts the number of flexes during an episode
        self.flex_counter = 0

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
        self.action_space = GlobalSpace([DiscreteSpace(2)])

        """
        SETTING THE OBSERVATION SPACE

        Set in the parent class BaseEnv
        """
        obs_spc, self.observation_ts, self.state_ts = self.get_state_obs()

        #  add infomation onto our observation
        obs_spc.extend([DiscreteSpace(1),
                        DiscreteSpace(self.flex_down_time),
                        DiscreteSpace(self.flex_up_time),
                        DiscreteSpace(self.relax_time)])

        self.observation_info.extend(['flex_availability',
                                      'flex_down_cycle',
                                      'flex_up_cycle',
                                      'relax_cycle'])

        self.observation_space = GlobalSpace(obs_spc)

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
        self.flex_avail = 1  # 0=not available, 1=available
        self.flex_down = 0
        self.flex_up = 0
        self.relax = 0

        #  initialize our action checker
        self.local_action = 0
        #  our env also keeps a list of the times when we started flexing
        self.flex_start_steps = []

        #  Resetting steps, state, observation, done status
        self.steps = 0
        self.state = self.get_state(steps=self.steps)
        self.observation = self.get_observation(self.steps,
                                                append=[self.flex_avail,
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
            observation (np.array) shape=(1, self.observation_space.shape[0])
            reward (float)
            done (bool)
            info (dict)

        Flex asset is dispatched if action=1 and not already in a flex cycle
        or relaxing.
        """
        #  pull the electricity price out of the state
        price_index = self.state_info.index('C_electricity_price_[$/MWh]')
        electricity_price = self.state[0][price_index]

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
            self.flex_avail = 1

        #  if we are not doing anything but want to start the flex down cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 1:
            self.flex_down = 1
            self.flex_avail = 0
            self.local_action = action
            self.flex_counter += 1

        #  if we are not doing anything but want to start the flex up cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 2:
            self.flex_up = 1
            self.flex_avail = 0
            self.local_action = action
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
        reward = flex_action * electricity_price / 12

        if flex_action == 0:
            flex_counter = 'not_flexing'
        else:
            flex_counter = 'flex_action_{}'.format(self.flex_counter)

        total_counters = self.check_counters()

        logger.debug('step {} elect. price {}'.format(
            self.observation_ts.index[self.steps], electricity_price))

        if total_counters > 0:
            logger.debug('action is {}'.format(action))
            logger.debug('flex_action is {}'.format(flex_action))
            logger.debug('up {} down {} relax {} rew {}'.format(
                self.flex_up, self.flex_down, self.relax, reward))

        self.steps += 1
        next_state = self.get_state(self.steps)
        next_observation = self.get_observation(self.steps,
                                                append=[self.flex_avail,
                                                        self.flex_down,
                                                        self.flex_up,
                                                        self.relax])

        #  check to see if we are done
        if self.steps == (self.episode_length - 1):
            done = True
        else:
            done = False

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
                                     flex_avail=self.flex_avail,
                                     flex_action=flex_action,
                                     flex_counter=flex_counter)

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, done, self.info

    def check_counters(self):
        """
        Helper function to check that the counters are OK
        """
        total_counters = sum([self.flex_up, self.flex_down, self.relax])

        if self.flex_avail == 0:
            assert total_counters != 0

        if self.flex_avail == 1:
            assert total_counters == 0

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

        return total_counters
