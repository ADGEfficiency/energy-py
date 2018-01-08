import os

from energy_py.agents import DispatchAgent, RandomAgent, NaiveFlex
from energy_py.envs import FlexEnv
from energy_py.experiments import no_learning_experiment

if __name__ == '__main__':

    data_path = os.getcwd()

    env = FlexEnv

    # dispatch = no_learning_experiment(DispatchAgent,
    #                                   env,
    #                                   data_path,
    #                                   'dispatch',
    #                                   opt_agent_args={'trigger':200})

    naive = no_learning_experiment(NaiveFlex,
                                   env,
                                   data_path,
                                   'naive_flex',
                                   opt_agent_args={'hours':[15]})

    # for j in range(2):
    #     random = no_learning_experiment(RandomAgent,
    #                                     env,
    #                                     data_path,
    #                                     'random_{}'.format(j),
    #                                     opt_agent_args={'sample_probability':0.1})
    
