import copy
import itertools
import os
import random

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import agents.value_functions
import assets.utils


#  agent based on Double Q-Learning
class agent(object):

    def __init__(self, env, verbose, device, run_name='TEST'):
        self.env = env
        self.verbose = verbose
        self.path = './results/'+run_name+'/'
        self.timer = assets.utils.Timer()

        input_length = len(self.env.s_mins) + len(self.env.a_mins)
        self.Q = self.get_value_functions(input_length, device)
        self.batch_size = 64  # size of batch for sampling memory
        self.epochs = 50
        self.memory, self.network_memory, self.info, self.age = self.get_memory()
        self.save_csv = False  # ???

        self.epsilon_decay = 0.9999  # decayed per policy decision
        self.policy_ = 0   # 0 = naive, 1 = e-greedy
        self.discount = 0.9  # discount factor for next_state
        self.test_state_actions = self.get_test_state_actions()

    def get_memory(self, default_epsilon=1):
        paths = ['memory.pickle',
                 'network_memory.pickle',
                 'info.pickle']
        objs = []
        for path in paths:
            if os.path.exists(self.path+path):
                print(path+' already exists')
                objs.append(assets.utils.load_pickle(self.path+path))
                self.age = objs[0][-1][10]
            else:
                print('Creating new memory for '+path)
                self.age = 0
                objs.append([])

        if self.age == 0:
            self.hists = [[],[],[]]
            self.epsilon = default_epsilon
        else:
            self.hists = objs[2][-1][3]
            self.epsilon = objs[0][-1][7]

        print('Age is {}.'.format(self.age))
        return objs[0], objs[1], objs[2], self.age

    def get_value_functions(self, input_length, device, value_functions=2):
        mdl_paths = [self.path + 'Q'+str(i)+'.h5' for i in range(value_functions)]
        Q = [None] * value_functions

        for j, path in enumerate(mdl_paths):
            if os.path.exists(path):
                print('Q{} function already exists'.format(j+1))
                Q[j] = keras.models.load_model(path)
            else:
                print('Q{} function being created'.format(j+1))
                Q[j] = agents.value_functions.Dense_Q(input_length, device=device)
        return Q


    def save_agent_brain(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.Q[0].save(filepath=self.path+'Q1.h5')
        self.Q[1].save(filepath=self.path+'Q2.h5')
        print('saved Q functions')
        assets.utils.dump_pickle(self.memory, self.path+'memory.pickle')
        assets.utils.dump_pickle(self.network_memory, self.path+'network_memory.pickle')
        assets.utils.dump_pickle(self.info, self.path+'info.pickle')
        print('saved memories')
        return self

    def single_episode(self, episode_number):
        print('Starting episode ' + str(episode_number))
        done = False
        self.age += 1
        state = self.env.reset()
        while done is not True:
            state = self.env.state
            action, state_action, choice = self.policy(state)
            next_state, reward, done, env_info = self.env.step(action)

            # training on non-NAIVE episodes
            if self.policy_ > 0:
                self.hists = self.train_model()

            self.memory.append([copy.copy(state),
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

            self.network_memory.append([copy.copy(self.normalize([state_action])),
                                        copy.copy(reward),
                                        copy.copy(self.state_to_state_actions(next_state)[1]),
                                        copy.copy(done)])

            self.info.append([self.timer.get_time(),
                              np.mean(self.Q[0].predict(self.test_state_actions)),
                              np.mean(self.Q[1].predict(self.test_state_actions)),
                              copy.copy(self.hists)])

            if self.verbose > 0:
                print('episode {} - step {} - choice {}'.format(episode_number, self.env.steps, choice))
                print('epsilon is {}'.format(self.epsilon))
                print('age is {}'.format(self.age))

        self.save_agent_brain()
        print('Finished episode {}.'.format(episode_number))
        print('Age is {}.'.format(self.age))
        print('Total run time is {}.'.format(self.timer.get_time()))
        return self


    def policy(self, state):
        state = state.values

        if self.policy_ == 0:  # naive
            choice = 'NAIVE'
            action = [action_space.high for action_space in self.env.action_space]

        elif self.policy_ == 1:  # e-greedy
            if random.random() < self.epsilon:  # exploring
                choice = 'RANDOM'
                action = [np.random.choice(np.array(action_space.sample()).flatten())
                          for action_space in self.env.action_space]
            else:  # acting according to Q1 & Q2
                choice = 'GREEDY'
                # we now use both of our Qfctns to select an action
                state_actions, norm_state_actions = self.state_to_state_actions(state)  # array of shape (num state actions, state_action dim)
                both_returns = [Qfctn.predict(norm_state_actions) for Qfctn in self.Q] # uses both Q fctns
                returns = np.add(both_returns[0], both_returns[1]) / 2

                idx = np.argmax(returns)
                optimal_state_action = state_actions[idx]
                optimal_action = optimal_state_action[len(self.env.state):]

                both_estimates = [rtn[idx] for rtn in both_returns]
                print('Q1 estimate={} - Q2 estimate={}.'.format(both_estimates[0], both_estimates[1]))

                state_action = optimal_action
                action = optimal_action

        # decaying epsilon
        self.epsilon = self.decay_epsilon()

        action = np.array(action).reshape(-1)
        state_action = np.concatenate([state, action])
        return action, state_action, choice

    def state_to_state_actions(self, state):
        action_space = self.env.create_action_space()
        bounds = [np.arange(asset.low, asset.high + 1) for asset in action_space]
        actions = [np.array(tup) for tup in list(itertools.product(*bounds))]
        state_actions = [np.concatenate((state, a)) for a in actions]
        norm_state_actions = np.vstack(self.normalize(state_actions))
        return state_actions, norm_state_actions

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

    def train_model(self, memory_length=50000):
        if self.verbose > 0:
            print('Starting training')

        # setting our Q functions
        REVERSE = random.choice([True, False])
        if REVERSE:  # randomly swapping which Q we use
            Q1 = self.Q[1]
            Q2 = self.Q[0]
            print('training model Q[0] using prediction from Q[1]')
        else:
            Q1 = self.Q[0]
            Q2 = self.Q[1]
            print('training model Q[1] using prediction from Q[0]')


        sample_size = min(len(self.network_memory), self.batch_size)
        memory = random.sample(self.network_memory[-memory_length:], sample_size)
        batch = np.array(memory)

        X = np.hstack(batch[:, 0]).reshape(sample_size, -1)  # state_actions
        reward = batch[:, 1]
        next_state_actions = batch[:, 2]

        # taking advantage of constant action space size here
        num_state_actions = batch.shape[0] * next_state_actions[0].shape[0]
        unstacked = np.vstack(next_state_actions).reshape(num_state_actions, -1)  # shape = (num_state_actions, state_action_length)
        predictions = self.discount * Q1.predict(unstacked) # shape = (num_state_actions, 1)
        predictions = predictions.reshape(sample_size, # shape = (batch_size,
                                          next_state_actions[0].shape[0], # number of state actions
                                          -1)  # predicted Q1(s,a)

        maximum_returns = np.amax(predictions, 1).reshape(-1)  # shape = (max[Q1(s,a)],)
        Y = np.add(reward, maximum_returns)

        if self.verbose > 0:
            print('Fitting model')

        # fiting model Q2 using predictions from Q1
        hist = Q2.fit(X, Y, epochs=self.epochs, batch_size=sample_size, verbose=self.verbose)

        if REVERSE:
            self.Q[0] = Q2
            self.Q[1] = Q1
            self.hists[0] += hist.history['loss']
            self.hists[1] += [0] * self.epochs
        else:
            self.Q[0] = Q1
            self.Q[1] = Q2
            self.hists[0] += [0] * self.epochs
            self.hists[1] += hist.history['loss']

        self.hists[2] += hist.history['loss']

        return self.hists
        
    def get_test_state_actions(self):
        Q_test = self.env.get_test_state_actions()
        test_state_actions = np.array(Q_test.values)
        test_state_actions = self.normalize(test_state_actions)
        return test_state_actions


    def decay_epsilon(self):
        if self.epsilon != 0:
            self.epsilon = self.epsilon * self.epsilon_decay
        return self.epsilon


    def create_outputs(self):
        def fig_training_history(hists, agent_info):
            fig, ax = plt.subplots(2, 3, sharex=False)
            fig.set_size_inches(8*2, 6*2)
            hists.plot(y='Q1',
                       kind='line',
                       ax=ax[0,0],
                       use_index=True)
            ax[0,0].set_title('Q1 Training History All Time',  fontsize=18)

            agent_info.plot(y='Q1 Test',
                            kind='line',
                            ax=ax[1,0],
                            use_index=True)
            ax[1,0].set_title('Q1 Test All Time',  fontsize=18)

            hists.plot(y='Q2',
                       kind='line',
                       ax=ax[0,1],
                       use_index=True)
            ax[0,1].set_title('Q2 Training History All Time',  fontsize=18)

            agent_info.plot(y='Q2 Test',
                            kind='line',
                            ax=ax[1,1],
                            use_index=True)
            ax[1,1].set_title('Q2 Test All Time',  fontsize=18)

            hists.plot(y='Q1 + Q2',
                       kind='line',
                       ax=ax[0,2],
                       use_index=True)
            ax[0,2].set_title('Q1 + Q2 Training History All Time',  fontsize=18)

            fig.savefig(self.path+'figures/training_history.png')
            return fig

        def fig_episode_return(summary):
            fig, ax = plt.subplots(2, 1, sharex=False)
            fig.set_size_inches(8*2, 6*2)

            ax[0].plot(summary.index,
                          summary.loc[:, 'Total Reward'],
                          label='Total Reward',
                          marker='o')
            ax[0].set_title('Total Reward All Time')

            fig.savefig(self.path+'figures/episode_return.png')
            return fig

        print('Generating outputs')
        plt.style.use('seaborn-deep')

        memory = pd.DataFrame(self.memory,
                              columns=['State', 'Action', 'State Action',
                                       'Reward', 'Next State', 'Episode',
                                       'Step', 'Epsilon', 'Choice', 'Done', 'Age'])

        agent_info = pd.DataFrame(self.info,
                                  columns=['Run Time', 'Q1 Test', 'Q2 Test', 'Hists'])

        network_memory = pd.DataFrame(self.network_memory,
                                      columns=['State Action (normalized)',
                                               'Reward', 'Next State & Actions', 'Done'])

        sums = memory.groupby(['Age']).sum().add_prefix('Total ')
        means = memory.groupby(['Age']).mean().add_prefix('Average ')
        summary = pd.concat([sums, means], axis=1)
        summary.loc[:, 'Maximum Reward'] = summary.loc[:, 'Total Reward'].cummax(axis=0)

        memory = pd.concat([memory, agent_info], axis=1)
        hists = pd.DataFrame(data={'Q1':self.hists[0],
                                   'Q2':self.hists[1],
                                   'Q1 + Q2':self.hists[2]})

        f1 = fig_training_history(hists, agent_info)
        f2 = fig_episode_return(summary)
        env_info = self.env.create_outputs(path=self.path)
        return summary
