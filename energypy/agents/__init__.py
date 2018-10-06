import logging

from energypy.agents.dqn import DQN
from energypy.agents.naive import NoOp, RandomAgent


logger = logging.getLogger(__name__)


agent_register = {
    'dqn': DQN,
    'random': RandomAgent,
    'no_op': NoOp,
}


def make_agent(agent_id, **kwargs):
    """ initializes an agent """
    logger.info('Making agent {}'.format(agent_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    return agent_register[agent_id](**kwargs)
