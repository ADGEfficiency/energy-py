from energy_py.common.memories.array_memory import ArrayMemory
from energy_py.common.memories.deque_memory import DequeMemory
from energy_py.common.memories.prioritized_replay import PrioritizedReplay

memory_register = {
    'array': ArrayMemory,
    'deque': DequeMemory,
    'prioritized': PrioritizedReplay
}
