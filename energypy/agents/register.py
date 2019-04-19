""" register for energypy agents """

from energypy.agents.dqn import DQN
from energypy.agents.naive import NoOp, RandomAgent


agent_register = {
    'dqn': DQN,
    'random': RandomAgent,
    'no_op': NoOp,
}


def make_agent(agent_id, **kwargs):
    """ grabs class from register and initializes """
    return agent_register[agent_id](**kwargs)
