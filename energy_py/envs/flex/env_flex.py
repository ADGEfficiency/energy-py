import logging
import os

from energy_py import DiscreteSpace, GlobalSpace
from energy_py.envs import TimeSeriesEnv


class FlexEnv(TimeSeriesEnv):
    def __init__(self,
                 episode_length='maximum',
                 episode_start=0,
                 flex_size=2,  # MW
                 flex_time=6,  # num 5 minute periods
                 relax_time=12):  # num 5 min periods

        data_path = os.path.dirname(os.path.abspath(__file__))

        #  technical energy inputs
        self.flex_size = float(flex_size)
        self.flex_time = int(flex_time)
        self.relax_time = int(relax_time)

        super().__init__(episode_length,
                         episode_start,
                         data_path)

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
        self.action_space = GlobalSpace(spaces=[DiscreteSpace(0, 1)])

        #  initialize all of the
        self.flex_avail = 1  # 0=not available, 1=available
        self.flex_up = 0
        self.flex_down = 0
        self.relax = 0

        """
        SETTING THE OBSERVATION SPACE

        Set in the parent class TimeSeriesEnv
        Append the flex asset availability to send to the agent
        TODO probably worth appending some more stuff as well!
        """
        obs_spc, self.observation_ts, self.state_ts = self.get_state_obs()

        #  grab the spaces list as setup by get_state_obs()
        #  add on a space to represent the flex availability
        spaces = obs_spc.spaces.append(DiscreteSpace(0, 1))
        self.observation_space = GlobalSpace(spaces)

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
        electricity_price = self.state[0][0]

        #  grab the action
        action = action[0][0]
        assert action >= self.action_space.spaces[0].low
        assert action <= self.action_space.spaces[0].high

        logging.info('electricity_price is {}'.format(electricity_price))
        logging.info('action is {}'.format(action))

        #  check counters are consistent
        self.check_counters()

        #  if we are in the flex up cycle, continue
        if self.flex_up > 0:
            self.flex_up += 1

        #  if we are in the flex down cycle, continue that
        if self.flex_down > 0:
            self.flex_down += 1

        #  if we are in the relaxation period, continue that
        if self.relax > 0:
            self.relax += 1

        #  if we are ending the flex up period, we start flex down
        if self.flex_up > self.flex_time:
            self.flex_up = 0
            self.flex_down = 1

        #  if we are ending the flex down period, we start relaxation
        if self.flex_down > self.flex_time:
            self.flex_down = 0
            self.relax = 1

        #  ending the relaxation period
        if self.relax > self.relax_time:
            self.relax = 0
            self.flex_avail = 1

        #  if we are not doing anything but want to start the flex cycle
        total_counters = sum([self.flex_up, self.flex_down, self.relax])
        if total_counters == 0 and action == 1:
            self.flex_up = 1
            self.flex_avail = 0

        #  we set the default action to do nothing
        flex_action = 0

        #  we set the flex action to do something if we are flexing
        #  up or down
        if self.flex_up > 0:
            flex_action = - self.flex_size

        if self.flex_down > 0:
            flex_action = self.flex_size

        #  now we set reward based on whether we are in a cycle or not
        #  reward is negative as we want to reduce cost
        reward = - flex_action * electricity_price

        logging.debug('flex_up {}'.format(self.flex_up))
        logging.debug('flex_down {}'.format(self.flex_down))
        logging.debug('relax {}'.format(self.relax))
        logging.debug('reward {}'.format(reward))

        self.steps += 1
        next_state = self.get_state(self.steps, append=self.flex_avail)
        next_observation = self.get_observation(self.steps, self.flex_avail)

        #  check to see if we are done
        if self.steps == (self.episode_length - 1):
            self.done = True

        self.info = self.update_info(episode=self.episode,
                                     steps=self.steps,
                                     state=self.state,
                                     observation=self.observation,
                                     action=action,
                                     reward=reward,
                                     next_state=next_state,
                                     next_observation=next_observation,
                                     done=self.done,

                                     electricity_price=electricity_price,
                                     flex_up=self.flex_up,
                                     flex_down=self.flex_down,
                                     relax=self.relax,
                                     flex_avail=self.flex_avail,
                                     flex_action=flex_action)

        #  moving to the next time step
        self.state = next_state
        self.observation = next_observation

        #  check counters are consistent
        self.check_counters()

        return self.observation, reward, self.done, self.info

    def _output_results(self):
        """
        Saves the state and observation dataframes to the outputs dict.
        Saves the environment panel figure args to the outputs dict.

        returns
            self.outputs (dict)
        """
        #  add the state and observation time series to the output dict
        self.outputs['state_ts'] = self.state_ts
        self.outputs['observation_ts'] = self.observation_ts

        #  arguments for the Visualizer make_panel_fig() function
        env_panel_fig = {'name': 'last_ep',
                         'ylims': [[0, 6], [], [], []],
                         'kinds': ['line', 'line', 'line', 'line'],
                         'panels': [['flex_up', 'flex_down', 'relax'],
                                    ['flex_avail', 'flex_action'],
                                    ['electricity_price'],
                                    ['reward']],
                         'shape': (4, 1)}

        #  add to the outputs dictionary
        self.outputs['env_panel_fig'] = env_panel_fig

        return self.outputs

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
