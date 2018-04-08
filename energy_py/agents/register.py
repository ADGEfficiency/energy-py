"""
A registry for agents
"""

import logging

from energy_py.agents.rand import RandomAgent
from energy_py.agents.naive import NaiveBatteryAgent
from energy_py.agents.naive import DispatchAgent
from energy_py.agents.naive import NaiveFlex

from energy_py.agents.dqn import DQN, Qfunc


logger = logging.getLogger(__name__)

agent_register = {'DQN': DQN}


def make_agent(agent_id, **kwargs):

    logger.info('Making agent {}'.format(agent_id))

    [logger.info('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    agent = agent_register[agent_id]

    return agent(**kwargs)

