from energy_py.experiments import reinforce_experiment
from energy_py.envs import BatteryEnv

if __name__ == '__main__':
    env = BatteryEnv
    agent_outputs, env_outputs = reinforce_experiment(env)

