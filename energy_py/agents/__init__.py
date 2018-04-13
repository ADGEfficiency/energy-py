from energy_py.agents.memory import Memory, ArrayMemory, DequeMemory
from energy_py.agents.memory import Experience, calculate_returns
from energy_py.agents.priority_memory import PrioritizedReplay

#  memories goes here because it's needed by BaseAgent
memories = {'array': ArrayMemory,
            'deque': DequeMemory,
            'priority': PrioritizedReplay}

from energy_py.agents.agent import BaseAgent
from energy_py.agents.register import make_agent

from energy_py.agents.classifier_agent import ClassifierCondition, ClassifierStragety
from energy_py.agents.classifier_agent import ClassifierStragety
