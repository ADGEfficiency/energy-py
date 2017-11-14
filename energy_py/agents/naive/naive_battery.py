import numpy as np

from energy_py.agents import Base_Agent

class NaiveBatteryAgent(Base_Agent):
    """
    This naive agent takes actions used predefined rules.

    A naive agent is useful as a baseline for comparing with reinforcement
    learning agents.

    As the rules are predefined each agent is specific to an environment.

    This agent is designed to control the battery environment.
    """

    def __init__(self, env):
        #  calling init method of the parent Base_Agent class
        #  passing the environment to the Base_Agent
        super().__init__(env)

    def _reset(self):
        #  nothing additional to be reset for this agent
        return None

    def _act(self, observation, session=None, epsilon=None):
        """
        Agent recieves a numpy array as the observation

        Agent makes determinsitc actions based on the observation
        """

        #  extracting the info from the observation
        electricity_price  = observation[0]
        electricity_demand = observation[1]
        month              = observation[2]
        day                = observation[3]
        hour               = observation[4]
        minute             = observation[5]
        weekday            = observation[6]
        current_charge     = observation[7]

        #  simple rules to decide what actions to take
        if hour >= 6 and hour < 10:
            #  discharge at max rate
            action = [self.action_space[0].low, self.action_space[1].high]

        elif hour >= 15 and hour < 21:
            #  discharge at max rate
            action = [self.action_space[0].low, self.action_space[1].high]

        else:
            #  charge at max rate
            action = [self.action_space[0].high, self.action_space[1].low]

        return np.array(action)

    def _learn(self):
        print('I am an agent based on a human desgined heuristic')
        print('I cannot learn anything')
        return None

    def _load_brain(self):
        print('I am an agent based on a human desgined heuristic')
        print('I have no brain')
        #  TODO could get this to load a rule from disk
        return None
