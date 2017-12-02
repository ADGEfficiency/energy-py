from energy_py.experiments import reinforce_experiment
from energy_py.envs import FlexEnv

if __name__ == '__main__':
    env = FlexEnv
    agent_outputs, env_outputs = reinforce_experiment(env)

