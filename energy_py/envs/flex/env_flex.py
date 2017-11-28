
from energy_py.envs import DiscreteSpace, TimeSeriesEnv

class FlexEnv(TimeSeriesEnv):
    def __init__(self,
                 episode_length='maximum',
                 episode_start=0,
                 flex_size=2, # MW
                 flex_time=6, # num 5 minute periods
                 relax_time=288): # num 5 min periods

    path = os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.join(path, 'state.csv')
    observation_path = os.path.join(path, 'observation.csv')

    #  technical energy inputs
    self.flex_size = float(flex_size)
    self.flex_time = int(flex_time)
    self.relax_time = int(relax_time)

    super().__init__(episode_length,
                     episode_start,
                     state_path,
                     observation_path)

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
        self.action_space = DiscreteSpace(0, 1)

        self.flex_counter = 0 
        self.relax_counter = 0
        self.flex_avail = 1 # 0=not available, 1=available
        
        """
        SETTING THE OBSERVATION SPACE

        Set in the parent class TimeSeriesEnv
        Append the flex asset availability to send to the agent
        """
        self.observation_space.spaces.append(DiscreteSpace(0, 1))
        
        """
        Intentionally not choosing to set reward space. 
       
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
        assert action.shape == (1,1)
        action = action[0]

        electricity_price = self.state[0]

        if self.flex_counter > 0:
            self.flex_counter += 1

        if self.relax_counter > 0:
            self.relax_counter += 1

        if self.flex_counter > self.flex_time:
            self.flex_counter = 0
            self.relax_counter = 1

        if self.relax_counter > self.relax_time:
            self.relax_counter = 0

        if sum(self.flex_counter, self.relax_counter) > 0 and action == 1:
            self.flex_counter = 1
            self.flex_avail = 0

        #  now we set reward based on whether we are in a cycle or not
        if self.flex_counter > 0:
            reward = self.flex_size * electricity_price 

        else:
        #  sparse reward signal
            reward = 0

        #  check to see if we are done
        if self.steps == (self.episode_length - 1):
            self.done = True
            next_state = 'Terminal'
            next_observation = 'Terminal'
        
        else:
        #  if we aren't done we move to next state
            self.steps += 1
            next_state = self.get_state(self.steps, append=self.flex_avail)
            self.observation = self.get_observation(self.steps, self.flex_avail)

        self.info = self.update_info(episode=self.episode,
                                     steps=self.steps,
                                     state=self.state,
                                     observation=self.observation,
                                     action=action,
                                     reward=reward,
                                     next_state=next_state,
                                     next_observation=next_observation,

                                     electricity_price=electricity_price,
                                     flex_counter=self.flex_counter,
                                     relax_counter=self.relax_counter,
                                     flex_avail = self.flex_avail)

        #  moving to the next time step
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def _output_results(self):
        return self.outputs
