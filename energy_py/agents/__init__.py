import logging

# from energy_py.agents.dqn import DQN
from energy_py.agents.dqn import DQN
from energy_py.agents.classifier import ClassifierAgent
from energy_py.agents.naive import NaiveFlex, RandomAgent


logger = logging.getLogger(__name__)


agent_register = {
    'dqn': DQN,
    'classifier': ClassifierAgent,
    'random': RandomAgent,
    'naive_flex': NaiveFlex
}


def make_agent(agent_id, **kwargs):

    logger.info('Making agent {}'.format(agent_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    agent = agent_register[agent_id]

    return agent(**kwargs)

