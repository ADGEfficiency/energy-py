from array_memory import ArrayMemory
from deque_memory import DequeMemory
from prioritized_replay import PrioritizedReplay

memory_register = {
    'array': ArrayMemory,
    'deque': DequeMemory,
    'prioritized': PrioritizedReplay
}
