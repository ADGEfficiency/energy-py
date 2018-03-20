from energy_py.agents.memory import Memory, ArrayMemory, DequeMemory
from energy_py.agents.memory import Experience, calculate_returns
from energy_py.agents.priority_memory import PrioritizedReplay

memories = {'array': ArrayMemory,
            'deque': DequeMemory,
            'priority': PrioritizedReplay}

from energy_py.agents.agent import BaseAgent

from energy_py.agents.rand import RandomAgent
from energy_py.agents.naive import NaiveBatteryAgent, DispatchAgent, NaiveFlex

from energy_py.agents.dqn import DQN, Qfunc
from energy_py.agents.dpg import DPG
