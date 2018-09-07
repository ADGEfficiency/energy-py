from energy_py.common.policies.epsilon_greedy import epsilon_greedy_policy
from energy_py.common.policies.softmax import softmax_policy

policy_register = {
    'epsilon_greedy': epsilon_greedy_policy,
    'softmax': softmax_policy
}
