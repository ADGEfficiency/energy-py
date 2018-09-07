import logging

from energypy.agents.dqn import DQN
from energypy.agents.naive import TimeFlex, RandomAgent, AutoFlex, NoOp


logger = logging.getLogger(__name__)


agent_register = {
    'dqn': DQN,
    'random': RandomAgent,
    'no_op': NoOp,
    'timeflex': TimeFlex,
    'autoflex': AutoFlex
}


def make_agent(agent_id, **kwargs):

    logger.info('Making agent {}'.format(agent_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    agent = agent_register[agent_id]

    return agent(**kwargs)
