from epsilon_greedy import epsilon_greedy_policy
from softmax import softmax_policy

policy_register = {
    'epsilon_greedy': epsilon_greedy_policy,
    'softmax': softmax_policy
}
