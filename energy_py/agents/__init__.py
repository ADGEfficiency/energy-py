from energy_py.agents.memory import ArrayMemory, DequeMemory

memories = {'array': ArrayMemory,
            'deque': DequeMemory}

from energy_py.agents.agent import BaseAgent, EpsilonGreedy

from energy_py.agents.rand import RandomAgent
from energy_py.agents.naive import NaiveBatteryAgent, DispatchAgent, NaiveFlex, ClassifierAgent

from energy_py.agents.reinforce import REINFORCE
from energy_py.agents.dqn import DQN, Qfunc
from energy_py.agents.dpg import DPG
