from energypy.common.memories.array_memory import ArrayMemory
from energypy.common.memories.deque_memory import DequeMemory

from energypy.common.utils import load_pickle


memory_register = {
    'array': ArrayMemory,
    'deque': DequeMemory,
}


def make_memory(**kwargs):
    """ makes a memory for an agent to store experience """

    load_path = kwargs.pop('load_path', None)

    if load_path:
        return load_pickle(load_path)

    else:
        memory_id = kwargs.pop('memory_id')
        memory = memory_register[memory_id]
        return memory(**kwargs)
