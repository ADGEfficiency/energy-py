import numpy as np

from energy_py.agents.core_agent import Base_Agent

class Naive_Battery_Agent(Base_Agent):
    """
    This naive agent takes actions used predefined rules.

    A naive agent is useful as a baseline for comparing with reinforcement
    learning agents.

    As the rules are predefined each agent is specific to an environment.

    This agent is designed to control the battery environment.
    """

    def __init__(self, env):
        #  calling init method of the parent Base_Agent class
        super().__init__(env)

    def _act(self, observation):
        """
        Agent recieves a numpy array as the observation.
        Agent makes decisions purely based the observation.
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
        if hour > 9 and hour < 11:
            #  discharge at max rate
            action = self.action_space[0].low

        elif hour > 17 and hour < 19:
            #  discharge at max rate
            action = self.action_space[0].low

        else:
            #  charge at max rate
            action = self.action_space[0].high
        print('hour was {} action was {}'.format(hour, action))
        return np.array([action])

    def _learn(self):
        print('I am an agent based on a human desgined heuristic.')
        print('I cannot learn anything.')
        return None

    def load_brain(self):
        print('I am an agent based on a human desgined heuristic.')
        print('I have no brain.')
        return None

if __name__ == '__main__':
    agent = Naive_Battery_Agent(env=1)
