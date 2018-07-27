"""
test - check that the step gives the correct one, test by running for n steps and comparing with the indexing method
"""
import energy_py
import numpy as np

env = energy_py.make_env('flex')

def get_state_actions(state):

    actions = env.action_space.discretize(3)

    return [np.concatenate([state.reshape(-1), action.reshape(-1)])
            for action in actions]

step = 4
state = env.observation_space.data.iloc[step, :]

state_actions = get_state_actions(state)

# def get_viable_transitions(step, state, actions):















