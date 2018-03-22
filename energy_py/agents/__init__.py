from energy_py.agents.memory import Memory, ArrayMemory, DequeMemory
from energy_py.agents.memory import Experience, calculate_returns
from energy_py.agents.priority_memory import PrioritizedReplay

#  memories goes here because it's needed by BaseAgent
memories = {'array': ArrayMemory,
            'deque': DequeMemory,
            'priority': PrioritizedReplay}

from energy_py.agents.agent import BaseAgent

from energy_py.agents.rand import RandomAgent
from energy_py.agents.naive import NaiveBatteryAgent
from energy_py.agents.naive import DispatchAgent
from energy_py.agents.naive import NaiveFlex

from energy_py.agents.dqn import DQN, Qfunc
from energy_py.agents.dpg import DPG

from energy_py.agents.classifier_agent import ClassifierCondition
from energy_py.agents.classifier_agent import ClassifierStragety
from energy_py.agents.classifier_agent import ClassifierAgent

