
from energy_py.agents.agent_core import Base_Agent

class Q_Learner(Base_Agent):

        def __init__(self, env,
                           epsilon_decay_steps,
                           batch_size    = 64):

        #  passing the environment to the Base_Agent class
        super().__init__(env, epsilon_decay_steps)
