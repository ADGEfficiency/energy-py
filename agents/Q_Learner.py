import copy
import itertools
import os
import random

import keras.models
import numpy as np
import pandas as pd

import assets.value_functions
import assets.utils


# agent based on Q-learning
class agent(object):

    def __init__(self, env, verbose, device):
        self.timer = assets.utils.Timer()
        self.verbose = verbose
        self.env = env

        input_length = len(self.env.s_mins) + len(self.env.a_mins)
        self.Q_path = './results/agent/EPISODE_LENGTH_'+str(self.env.episode_length)+'/'
        self.Q = self.get_value_function(input_length, device)
        self.batch_size = 64  # size of batch for sampling memory
        self.epochs = 50
        self.memory, self.network_memory, self.info, self.age = self.get_memory(self.Q_path)
        self.save_csv = False  # ???

        self.epsilon = 1.0  # initial exploration factor
        self.policy_ = 0   # 0 = naive, 1 = e-greedy
        self.discount = 0.9  # discount factor for next_state
        self.test_state_actions = self.get_test_state_actions()

    def get_memory(self, Q_path):
        paths = [Q_path+'/memory.pickle',
                 Q_path+'/network_memory.pickle',
                 Q_path+'/info.pickle']
        objs = []
        for path in paths:
            if os.path.exists(path):
                print('Memory already exists')
                objs.append(assets.utils.load_pickle(path))
                self.age = objs[0][-1][10]
            else:
                print('Creating new memory')
                self.age = 0
                objs.append([])

        print('Age is {}.'.format(self.age))
        return objs[0], objs[1], objs[2], self.age

    def get_value_function(self, input_length, device):
        # check to see whether model already exists
        mdl_path = self.Q_path + '/qfctn.h5'
        if os.path.exists(mdl_path):
            print('Q function already exists')
            Q = keras.models.load_model(mdl_path)
        else:
            print('creating new Q function')
            Q = assets.value_functions.Dense_Q(input_length, device=device)
        return Q

    def single_episode(self, episode_number):
        print('Starting episode ' + str(episode_number))
        state = self.env.reset()
        done = False
        self.age += 1
        final_loss = 0
        while done is not True:
            state = self.env.state
            action, state_action, choice = self.policy(state)
            next_state, reward, done, env_info = self.env.step(action)

            self.memory.append([
                copy.copy(state),
                copy.copy(action),
                copy.copy(state_action),
                copy.copy(reward),
                copy.copy(next_state),
                copy.copy(episode_number),
                copy.copy(self.env.steps),
                copy.copy(self.epsilon),
                copy.copy(choice),
                copy.copy(done),
                copy.copy(self.age)])

            self.network_memory.append([
                copy.copy(self.normalize([state_action])),
                copy.copy(reward),
                self.state_to_state_actions(next_state),
                copy.copy(done)])

            if self.policy_ > 0:
                hist = self.train_model()
                final_loss = hist.history['loss'][-1]
                self.epsilon = self.decay_epsilon(episode_number)

            self.info.append([
                self.timer.get_time(),
                np.mean(self.Q.predict(self.test_state_actions)),
                final_loss])

            if self.verbose > 0:
                print('episode {} - step {} - choice {}.'.format(episode_number, self.env.steps, choice))
                print('state {}.'.format(state.values))
                print('action {}.'.format(action))
                print('current asset state is {}.'.format(self.env.last_actions))
                print('state action {}.'.format(state_action))
                print('next state {}.'.format(next_state.values))
                print('reward {} - epsilon {} - step {}.'.format(reward, self.epsilon, self.env.steps))
                print('age is {}.'.format(self.age))


        self.save_agent_brain(self.Q_path)
        print('Finished episode {}.'.format(episode_number))
        print('Age is {}.'.format(self.age))
        print('Total run time is {}.'.format(self.timer.get_time()))
        return self

    def save_agent_brain(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.Q.save(filepath=self.Q_path+'Qfctn.h5')
        assets.utils.dump_pickle(self.memory, self.Q_path+'memory.pickle')
        assets.utils.dump_pickle(self.network_memory, self.Q_path+'network_memory.pickle')
        assets.utils.dump_pickle(self.info, self.Q_path+'info.pickle')

    def decay_epsilon(self, episode_number):
        # TODO harcoded to be at 0.1 after 25 episodes
        if self.epsilon != 0:
            self.epsilon = max(0.1, -0.0375 * episode_number + 1.0375)
        return self.epsilon

    def policy(self, state):
        state = state.values
        if self.policy_ == 0:  # naive
            choice = 'NAIVE'
            action = [action_space.high for action_space in self.env.action_space]
        elif self.policy_ == 1:  # e-greedy
            if random.random() < self.epsilon:
                choice = 'RANDOM'
                action = [np.random.choice(np.array(action_space.sample()).flatten())
                          for action_space in self.env.action_space]
            else:
                choice = 'GREEDY'
                state_actions = self.state_to_state_actions(state)
                v_stack = np.vstack(state_actions)
                returns = self.Q.predict(v_stack)
                optimal_state_action = list(state_actions[np.argmax(returns)].flatten())
                optimal_action = optimal_state_action[len(self.env.state):]
                normalized_action = copy.copy(optimal_action)
                lb, ub = self.env.a_mins, self.env.a_maxs
                denormalized_action = [
                    lb[i] + act * (ub[i] - lb[i])
                    for i, act in enumerate(normalized_action)
                    ]
                action = denormalized_action
                action = [int(act) for act in action]
        action = np.array(action).reshape(-1)
        state_action = np.concatenate([state, action])
        return action, state_action, choice

    def state_to_state_actions(self, state):
        action_space = self.env.create_action_space()
        bounds = []
        for asset in action_space:
            try:
                inner_bounds = []
                for action in asset.spaces:
                    rng = np.linspace(start=action.low,
                                      stop=action.high,
                                      num=(action.high - action.low) + 1)
                    inner_bounds.append(rng)
                inner_bounds = np.concatenate(inner_bounds)
                bounds.append(inner_bounds)
            except AttributeError:  # catches case that isn't Tuple
                rng = np.linspace(start=asset.low,
                                  stop=asset.high,
                                  num=(asset.high - asset.low) + 1)
                bounds.append(rng)

        actions = [np.array(tup) for tup in list(itertools.product(*bounds))]
        state_actions = [np.concatenate((state, a)) for a in actions]
        norm_state_actions = self.normalize(state_actions)
        return norm_state_actions

    def normalize(self, state_actions):
        mins, maxs = list(self.env.mins), list(self.env.maxs)
        norm_state_action, norm_state_actions = [], []
        for state_action in state_actions:
            length = len(state_action)

            for j, variable in enumerate(state_action):
                lb, ub = mins[j], maxs[j]
                normalized = (variable - lb) / (ub - lb)
                norm_state_action.append(normalized)

            norm_array = np.array(norm_state_action).reshape(-1, length)
            norm_state_actions.append(norm_array)
            norm_state_action = []
        norm_state_actions = np.array(norm_state_actions).reshape(-1,length)
        return norm_state_actions

    def train_model(self):
        if self.verbose > 0:
            print('Starting training')
        sample_size = min(len(self.network_memory), self.batch_size)
        memory_length = -50000
        batch = np.array(random.sample(self.network_memory[memory_length:],
                                       sample_size))
        features = np.hstack(batch[:, 0]).reshape(sample_size, -1)
        reward = batch[:, 1]
        next_state_actions = batch[:, 2]

        lengths = [np.vstack(item).shape[0] for item in next_state_actions]
        total_length = np.sum(lengths)
        start, stop = np.zeros(shape=len(lengths)), np.zeros(shape=len(lengths))
        unstacked = np.vstack(next_state_actions).reshape(total_length, -1)
        next_state_pred, num_not_unique = self.train_on_uniques(sa=unstacked)

        start, returns = 0, []
        for k in range(0, sample_size):
            stop = start + lengths[k]
            if batch[k, 3] is True:  # if last step
                rtn = 0
            else:
                rtn = np.max(next_state_pred[start:stop])
            start = stop
            returns.append(rtn)
        target = reward + returns
        X = features
        Y = target
        if self.verbose > 0:
            print('Fitting model')
        hist = self.Q.fit(
            X, Y, epochs=self.epochs, batch_size=sample_size, verbose=self.verbose
            )

        return hist

    def train_on_uniques(self, sa):
        b = np.ascontiguousarray(sa).view(np.dtype(
            (np.void, sa.dtype.itemsize * sa.shape[1])))
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
        uniques = sa[idx]
        unique_predictions = self.discount * self.Q.predict(uniques)
        all_preds = unique_predictions[inv]
        pct_not_unique = 100 * (sa.shape[0] - uniques.shape[0]) / sa.shape[0]
        if self.verbose == 1:
            print('number of state actions ' + str(sa.shape[0]))
            print('number of unique ' + str(uniques.shape[0]))
            print('Pct not unique was {0:.0f}%'.format(pct_not_unique))
        return all_preds, pct_not_unique

    def get_test_state_actions(self):
        Q_test = pd.read_csv('assets/Q_test.csv', index_col=[0])
        Q_test.iloc[:, 1:] = Q_test.iloc[:, 1:].apply(pd.to_numeric)
        test_state_actions = np.array(Q_test.iloc[:, 1:])
        test_state_actions = self.normalize(test_state_actions)
        return test_state_actions

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                DOUBLE Q LEARNER
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class Double_Q_learner(object):

    def __init__(self, env, verbose, device):
        self.timer = assets.utils.Timer()
        self.verbose = verbose
        self.env = env

        input_length = len(self.env.s_mins) + len(self.env.a_mins)
        self.Q_path = './results/agent/EPISODE_LENGTH_'+str(self.env.episode_length)+'/'
        self.Q = self.get_value_function(input_length, device)
        self.batch_size = 64  # size of batch for sampling memory
        self.epochs = 50
        self.memory, self.network_memory, self.info, self.age = self.get_memory(self.Q_path)
        self.save_csv = False  # ???

        self.epsilon = 1.0  # initial exploration factor
        self.policy_ = 0   # 0 = naive, 1 = e-greedy
        self.discount = 0.9  # discount factor for next_state
        self.test_state_actions = self.get_test_state_actions()

    def get_memory(self, Q_path):
        paths = [Q_path+'/memory.pickle',
                 Q_path+'/network_memory.pickle',
                 Q_path+'/info.pickle']
        objs = []
        for path in paths:
            if os.path.exists(path):
                print('Memory already exists')
                objs.append(assets.utils.load_pickle(path))
                self.age = objs[0][-1][10]
            else:
                print('Creating new memory')
                self.age = 0
                objs.append([])

        print('Age is {}.'.format(self.age))
        return objs[0], objs[1], objs[2], self.age

    def get_value_function(self, input_length, device):
        # check to see whether model already exists
        mdl_path = self.Q_path + '/qfctn.h5'
        if os.path.exists(mdl_path):
            print('Q function already exists')
            Q = keras.models.load_model(mdl_path)
        else:
            print('creating new Q function')
            Q = assets.value_functions.Dense_Q(input_length, device=device)
        return Q

    def single_episode(self, episode_number):
        print('Starting episode ' + str(episode_number))
        state = self.env.reset()
        done = False
        self.age += 1
        final_loss = 0
        while done is not True:
            state = self.env.state
            action, state_action, choice = self.policy(state)
            next_state, reward, done, env_info = self.env.step(action)

            self.memory.append([
                copy.copy(state),
                copy.copy(action),
                copy.copy(state_action),
                copy.copy(reward),
                copy.copy(next_state),
                copy.copy(episode_number),
                copy.copy(self.env.steps),
                copy.copy(self.epsilon),
                copy.copy(choice),
                copy.copy(done),
                copy.copy(self.age)])

            self.network_memory.append([
                copy.copy(self.normalize([state_action])),
                copy.copy(reward),
                self.state_to_state_actions(next_state),
                copy.copy(done)])

            # training on non-NAIVE episodes
            if self.policy_ > 0:
                hist = self.train_model()
                final_loss = hist.history['loss'][-1]
                self.epsilon = self.decay_epsilon(episode_number)

            self.info.append([
                self.timer.get_time(),
                np.mean(self.Q.predict(self.test_state_actions)),
                final_loss])

            if self.verbose > 0:
                print('episode {} - step {} - choice {}.'.format(episode_number, self.env.steps, choice))
                print('state {}.'.format(state.values))
                print('action {}.'.format(action))
                print('current asset state is {}.'.format(self.env.last_actions))
                print('state action {}.'.format(state_action))
                print('next state {}.'.format(next_state.values))
                print('reward {} - epsilon {} - step {}.'.format(reward, self.epsilon, self.env.steps))
                print('age is {}.'.format(self.age))


        self.save_agent_brain(self.Q_path)
        print('Finished episode {}.'.format(episode_number))
        print('Age is {}.'.format(self.age))
        print('Total run time is {}.'.format(self.timer.get_time()))
        return self

    def save_agent_brain(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.Q.save(filepath=self.Q_path+'Qfctn.h5')
        assets.utils.dump_pickle(self.memory, self.Q_path+'memory.pickle')
        assets.utils.dump_pickle(self.network_memory, self.Q_path+'network_memory.pickle')
        assets.utils.dump_pickle(self.info, self.Q_path+'info.pickle')

    def decay_epsilon(self, episode_number):
        # TODO harcoded to be at 0.1 after 25 episodes
        if self.epsilon != 0:
            self.epsilon = max(0.1, -0.0375 * episode_number + 1.0375)
        return self.epsilon

    def policy(self, state):
        state = state.values
        if self.policy_ == 0:  # naive
            choice = 'NAIVE'
            action = [action_space.high for action_space in self.env.action_space]
        elif self.policy_ == 1:  # e-greedy
            if random.random() < self.epsilon:
                choice = 'RANDOM'
                action = [np.random.choice(np.array(action_space.sample()).flatten())
                          for action_space in self.env.action_space]
            else:
                choice = 'GREEDY'
                state_actions = self.state_to_state_actions(state)
                v_stack = np.vstack(state_actions)
                returns = self.Q.predict(v_stack)
                optimal_state_action = list(state_actions[np.argmax(returns)].flatten())
                optimal_action = optimal_state_action[len(self.env.state):]
                normalized_action = copy.copy(optimal_action)
                lb, ub = self.env.a_mins, self.env.a_maxs
                denormalized_action = [
                    lb[i] + act * (ub[i] - lb[i])
                    for i, act in enumerate(normalized_action)
                    ]
                action = denormalized_action
                action = [int(act) for act in action]
        action = np.array(action).reshape(-1)
        state_action = np.concatenate([state, action])
        return action, state_action, choice

    def state_to_state_actions(self, state):
        action_space = self.env.create_action_space()
        bounds = []
        for asset in action_space:
            try:
                inner_bounds = []
                for action in asset.spaces:
                    rng = np.linspace(start=action.low,
                                      stop=action.high,
                                      num=(action.high - action.low) + 1)
                    inner_bounds.append(rng)
                inner_bounds = np.concatenate(inner_bounds)
                bounds.append(inner_bounds)
            except AttributeError:  # catches case that isn't Tuple
                rng = np.linspace(start=asset.low,
                                  stop=asset.high,
                                  num=(asset.high - asset.low) + 1)
                bounds.append(rng)

        actions = [np.array(tup) for tup in list(itertools.product(*bounds))]
        state_actions = [np.concatenate((state, a)) for a in actions]
        norm_state_actions = self.normalize(state_actions)
        return norm_state_actions

    def normalize(self, state_actions):
        mins, maxs = list(self.env.mins), list(self.env.maxs)
        norm_state_action, norm_state_actions = [], []
        for state_action in state_actions:
            length = len(state_action)

            for j, variable in enumerate(state_action):
                lb, ub = mins[j], maxs[j]
                normalized = (variable - lb) / (ub - lb)
                norm_state_action.append(normalized)

            norm_array = np.array(norm_state_action).reshape(-1, length)
            norm_state_actions.append(norm_array)
            norm_state_action = []
        norm_state_actions = np.array(norm_state_actions).reshape(-1,length)
        return norm_state_actions

    def train_model(self):
        if self.verbose > 0:
            print('Starting training')
        sample_size = min(len(self.network_memory), self.batch_size)
        memory_length = -50000
        batch = np.array(random.sample(self.network_memory[memory_length:],
                                       sample_size))
        features = np.hstack(batch[:, 0]).reshape(sample_size, -1)
        reward = batch[:, 1]
        next_state_actions = batch[:, 2]

        lengths = [np.vstack(item).shape[0] for item in next_state_actions]
        total_length = np.sum(lengths)
        start, stop = np.zeros(shape=len(lengths)), np.zeros(shape=len(lengths))
        unstacked = np.vstack(next_state_actions).reshape(total_length, -1)
        next_state_pred, num_not_unique = self.train_on_uniques(sa=unstacked)

        start, returns = 0, []
        for k in range(0, sample_size):
            stop = start + lengths[k]
            if batch[k, 3] is True:  # if last step
                rtn = 0
            else:
                rtn = np.max(next_state_pred[start:stop])
            start = stop
            returns.append(rtn)
        target = reward + returns
        X = features
        Y = target
        if self.verbose > 0:
            print('Fitting model')
        hist = self.Q.fit(
            X, Y, epochs=self.epochs, batch_size=sample_size, verbose=self.verbose
            )

        return hist

    def train_on_uniques(self, sa):
        b = np.ascontiguousarray(sa).view(np.dtype(
            (np.void, sa.dtype.itemsize * sa.shape[1])))
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
        uniques = sa[idx]
        unique_predictions = self.discount * self.Q.predict(uniques)
        all_preds = unique_predictions[inv]
        pct_not_unique = 100 * (sa.shape[0] - uniques.shape[0]) / sa.shape[0]
        if self.verbose == 1:
            print('number of state actions ' + str(sa.shape[0]))
            print('number of unique ' + str(uniques.shape[0]))
            print('Pct not unique was {0:.0f}%'.format(pct_not_unique))
        return all_preds, pct_not_unique

    def get_test_state_actions(self):
        Q_test = pd.read_csv('assets/Q_test.csv', index_col=[0])
        Q_test.iloc[:, 1:] = Q_test.iloc[:, 1:].apply(pd.to_numeric)
        test_state_actions = np.array(Q_test.iloc[:, 1:])
        test_state_actions = self.normalize(test_state_actions)
        return test_state_actions
