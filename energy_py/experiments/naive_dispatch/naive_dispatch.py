import os

from energy_py.agents import DispatchAgent, RandomAgent
from energy_py.envs import FlexEnv
from energy_py.experiments import no_learning_experiment

if __name__ == '__main__':

    data_path = os.getcwd()

    env = FlexEnv

    random = no_learning_experiment(RandomAgent,
                                    env,
                                    data_path,
                                    'no_op',
                                    opt_agent_args={'sample_probability':0.0})

    for j in range(2):
        random = no_learning_experiment(RandomAgent,
                                        env,
                                        data_path,
                                        'random_{}'.format(j),
                                        opt_agent_args={'sample_probability':0.1})

    
    dispatch = no_learning_experiment(DispatchAgent,
                                      env,
                                      data_path,
                                      'dispatch',
                                      opt_agent_args={'trigger':200})

