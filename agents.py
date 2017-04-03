import random
import itertools
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import assets.value_functions
import assets.utils


class Q_learner(object):

    def __init__(self, env, verbose):
        self.timer = assets.utils.Timer()
        self.verbose = verbose
        self.env = env
        input_length = len(self.env.s_mins) + len(self.env.a_mins)
        self.network = assets.value_functions.Dense_Q(input_length)
        self.batch_size = 64
        self.epochs = 100

        self.memory, self.network_memory, self.info = [], [], []
        self.save_csv = False

        self.epsilon = 1.0
        self.epsilon_decay = 0.9999 # TODO this calculated automagically
        self.policy_ = 0  # 0 = naive, 1 = e-greedy
        self.discount = 0.9
        self.test_state_actions = self.get_test_state_actions()

    def single_episode(self, episode_number):
        print('Starting episode ' + str(episode_number))
        state = self.env.reset()
        done = False
        while done is not True:
            state = copy.copy(self.env.state)
            last_action = copy.copy(self.env.last_actions)
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
                copy.copy(done)])

            self.network_memory.append([
                copy.copy(self.normalize([state_action])),
                copy.copy(reward),
                self.state_to_state_actions(next_state, action),
                copy.copy(done)])

            if episode_number == 0:
                final_loss = 0

            elif episode_number > 0:
                hist = self.train_model()
                final_loss = hist.history['loss'][-1]
                self.epsilon = self.decay_epsilon()

            self.info.append([
                self.timer.get_time(),
                np.mean(self.network.predict(self.test_state_actions)),
                final_loss])

            if self.verbose > 0:
                print('episode ' + str(episode_number) +
                      ' - step ' + str(self.env.steps) +
                      ' - choice ' + str(choice))
                print('state ' + str(state))
                print('last action was ' + str(last_action))
                print('action ' + str(action))
                print('state action ' + str(state_action))
                print('next state ' + str(next_state))
                print('reward ' + str(reward) +
                      ' - epsilon ' + str(self.epsilon))
        print('Finished episode ' + str(episode_number))
        print('Total run time is ' + self.timer.get_time())
        return self

    def decay_epsilon(self):
        if self.epsilon != 0:
            self.epsilon = max(0.1, self.epsilon_decay * self.epsilon)
        return self.epsilon

    def policy(self, state):
        if self.policy_ == 0:  # naive
            choice = 'NAIVE'
            action = self.env.action_space.high
        elif self.policy_ == 1:  # e-greedy
            if random.random() < self.epsilon:
                choice = 'RANDOM'
                action = self.env.action_space.sample()
            else:
                choice = 'GREEDY'
                state_actions = self.state_to_state_actions(
                    state,
                    self.env.last_actions
                    )
                v_stack = np.vstack(state_actions)
                returns = self.network.predict(v_stack)
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
        state_action = np.concatenate([state, action])
        return action, state_action, choice

    def state_to_state_actions(self, state, action):
        # use this fctn for both state (s) and next state (s')
        # correct action to pass is the last action
        action_space = self.env.create_action_space(action)
        a_bounds = [action_space.low, action_space.high]
        all_rng = []
        for m, act in enumerate(a_bounds[0]):
            rng = np.linspace(
                start=a_bounds[0][m],
                stop=a_bounds[1][m],
                num=(a_bounds[1][m] - a_bounds[0][m]) + 1
                )
            all_rng.append(rng)

        actions = [np.array(tup) for tup in list(itertools.product(*all_rng))]
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
        return norm_state_actions

    def train_model(self):
        if self.verbose > 0:
            print('Starting training')
        sample_size = min(len(self.network_memory), self.batch_size)
        batch = np.array(random.sample(self.network_memory, sample_size))
        # batch = np.array(batch)
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
            if batch[k, 3] == True:  # if last step
                rtn = 0
            else:
                rtn = np.max(next_state_pred[start:stop])
            start = stop
            returns.append(rtn)
        target = reward + returns
        X = features
        Y = target
        hist = self.network.fit(
            X, Y, epochs=self.epochs, batch_size=sample_size, verbose=self.verbose
            )

        return hist

    def train_on_uniques(self, sa):
        b = np.ascontiguousarray(sa).view(np.dtype(
            (np.void, sa.dtype.itemsize * sa.shape[1])))
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
        uniques = sa[idx]
        unique_predictions = self.discount * self.network.predict(uniques)
        all_preds = unique_predictions[inv]
        pct_not_unique = 100 * (sa.shape[0] - uniques.shape[0]) / sa.shape[0]
        if self.verbose == 1:
            print('number of state actions ' + str(sa.shape[0]))
            print('number of unique ' + str(uniques.shape[0]))
            print('Pct not unique was {0:.0f}%'.format(pct_not_unique))
        return all_preds, pct_not_unique

    def get_test_state_actions(self):
        test_state_actions = []
        for i in range(0, 4):
            state_action = np.concatenate([
                self.env.observation_space.sample(),
                self.env.action_space.sample()])
            test_state_actions.append(state_action)
        test_state_actions = np.array(test_state_actions)
        return test_state_actions

    def outputs(self):
        print('Generating outputs')
        plt.style.use('seaborn-deep')

        idx = np.linspace(1, len(self.memory), len(self.memory), endpoint=True)
        index = pd.Index(idx, dtype=int, name='Step')

        memory = pd.DataFrame(
            self.memory,
            index=index,
            columns=['State', 'Action', 'State Action', 'Reward', 'Next State',
                     'Episode', 'Step', 'Epsilon', 'Choice', 'Done']
            )

        agent_info = pd.DataFrame(
            self.info,
            index=index,
            columns=['Run Time', 'Q Test', 'Training History']
            )

        network_memory = pd.DataFrame(
            self.network_memory,
            index=index,
            columns=['State Action (normalized)',
                     'Reward', 'Next State & Actions','Done']
            )

        memory = pd.concat([memory, agent_info], axis=1)
        sums = memory.groupby(['Episode']).sum().add_prefix('Total ')
        means = memory.groupby(['Episode']).mean().add_prefix('Mean ')
        count = memory.groupby('Episode')['Step'].count().rename('Step')
        summary = pd.concat([sums, means, count], axis=1)

        x = summary.index
        steps = summary.loc[:, 'Step'].values
        tick_locations = np.cumsum(steps)
        last_step_start = int(tick_locations[-2])
        last_step_stop = int(tick_locations[-1])

        env_info = pd.DataFrame(
            self.env.info,
            index=range(last_step_start, last_step_stop),
            columns=['Settlement period',
                     'Power generated [MWe]',
                     'Import electricity price [£/MWh]',
                     'Total heat demand [MW]']
            )
        env_info.loc[:, 'Steps'] = np.arange(1, env_info.shape[0] + 1)

        # Figure 1
        f1, ax1 = plt.subplots(3, 1, sharex=True)
        sns.tsplot(memory.loc[:, 'Q Test'], memory.index, ax=ax1[0])
        sns.tsplot(memory.loc[:, 'Training History'], memory.index, ax=ax1[1])
        sns.tsplot(memory.loc[:, 'Training History'], memory.index, ax=ax1[2])
        ax1[0].set(ylabel='Q Test')
        ax1[1].set(ylabel='Training History')
        ax1[2].set(ylabel='Training History')
        ax1[2].set_ylim([0, 5000])
        ax2 = ax1[0].twiny()
        ax2.set_xlim(ax1[0].get_xlim())
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(summary.index)
        ax2.set_xlabel('Episode')
        f1.savefig('results/figures/f1.png')
        plt.close(f1)

        # Figure 2
        f2, ax2 = plt.subplots(2, 1, sharex=True)
        sns.tsplot(summary.loc[:, 'Mean Epsilon'], x, ax=ax2[0], marker='o')
        sns.tsplot(summary.loc[:, 'Total Reward'], x,
                   ax=ax2[1], marker='o', color='g')
        ax2[0].set(ylabel='Mean Epsilon')
        ax2[1].set(ylabel='Total Reward')
        f2.savefig('results/figures/f2.png')
        plt.close(f2)

        def col_to_graph(df, names):
            length = df.iloc[0, 0].shape[0]
            f, axes = plt.subplots(length, 1)
            for i, ax in enumerate(axes):
                df.loc[:, names[i]] = [item_list[i] for item_list in df.iloc[:, 0]]
                ax.plot(df.index, df.loc[:, names[i]])
                ax.set_xlim([last_step_start, last_step_stop])
                ax.set_ylabel(names[i], rotation=0)
            return f, df

        # Figure 3
        actions = memory.loc[:, 'Action'].to_frame(name='Actions')
        action_names = self.env.action_names
        f3, actions = col_to_graph(actions, action_names)
        f3.savefig('results/figures/f3.png')
        plt.close(f3)

        # Figure 4
        states = memory.loc[:, 'State'].to_frame(name='States')
        state_names = self.env.state_names
        f4, states = col_to_graph(states, state_names)
        f4.savefig('results/figures/f4.png')
        plt.close(f4)

        # Figure 5
        f5, ax5 = plt.subplots(1, 1)
        fig5_vars = ['Power generated [MWe]',
                     'Import electricity price [£/MWh]',
                     'Total heat demand [MW]']
        x_range = env_info['Steps'].as_matrix()
        for var in fig5_vars:
            ax5.plot(x_range, env_info.loc[:, var], label=var)
        ax5.legend(loc='best')
        ax5.set_xlabel('Steps')
        ax5.set_xlim(x_range[0], x_range[-1])
        ax5.set_xticks(x_range)
        f5.savefig('results/figures/f5.png')
        plt.close(f5)

        if self.save_csv:
            print('Saving csvs')
            memory.to_csv('results/memory.csv')
            network_memory.to_csv('results/network_memory.csv')
            summary.to_csv('results/summary.csv')
            states.to_csv('results/states.csv')
            actions.to_csv('results/actions.csv')
        print('Finished generating outputs')

        return memory, network_memory, summary, states, actions
