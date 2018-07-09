import logging

# from energy_py.agents.dqn import DQN
from energy_py.agents.dqn import DQN
from energy_py.agents.classifier import ClassifierAgent
from energy_py.agents.naive import TimeFlex, RandomAgent, AutoFlex, NoOp


logger = logging.getLogger(__name__)


agent_register = {
    'dqn': DQN,
    'classifier': ClassifierAgent,
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

