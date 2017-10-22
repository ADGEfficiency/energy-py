import numpy as np

from energy_py.agents import Base_Agent, Epsilon_Greedy


class DQN(Base_Agent):
    """
    energy_py implementation of DQN
    aka Q-learning with experience replay & target network

    args
        env                 : energy_py environment
        Q_actor             : energy_py action-value function
                              used for acting & learning
        Q_target            : energy_py action-value function
                              used for creating the target for learning
        epsilon_decay_steps : int

    inherits from
        Base_Agent          : the energy_py class used for agents

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self, env,
                       Q_actor,
                       Q_target,
                       discount,
                       epsilon_decay_steps=10000,
                       verbose=False):

        #  passing the environment to the Base_Agent class
        super().__init__(env, epsilon_decay_steps, verbose)

        self.Q_actor = Q_actor(self.observation_dim + self.num_actions)
        self.Q_target = Q_target(self.observation_dim + self.num_actions)

        self.discount = discount

        #  create an object to decay epsilon
        self.e_greedy = Epsilon_Greedy(decay_steps=epsilon_decay_steps,
                                       verbose=0)

    def _act(self, **kwargs):
        """
        Act using an epsilon-greedy policy

        args
            observation : np array (1, observation_dim)

        return
            action      : np array (1, num_actions)
        """
        #  pull out the relevant kwargs
        observation = kwargs.pop('observation')

        #  because our observation comes directly from the env
        #  we need to scale the observation
        observation = self.scale_array(observation, self.observation_space)

        #  get the current value of epsilon
        epsilon = self.e_greedy.get_epsilon()
        self.verbose_print('epsilon is {:.3f}'.format(epsilon))

        if np.random.uniform() < epsilon:
            self.verbose_print('acting randomly')
            action = [space.sample() for space in self.action_space]

        else:
            self.verbose_print('acting according to Q_actor')

            #  create all possible combinations of our single observation
            #  and our n-dimensional action space
            state_acts, acts = self.all_state_actions(self.action_space,
                                                   observation)

            #  get predictions from the action_value function Q
            Q_estimates = []
            for sa in state_acts:
                Q_estimates.append(self.Q_actor.predict(sa.reshape(1, -1)))

            #  select the action with the highest Q
            #  note that we index the unscaled action
            #  as this action is sent directly to the environment
            action = acts[np.argmax(Q_estimates)]

        action = np.array(action).reshape(1, self.num_actions)
        assert len(self.action_space) == action.shape[1]
        return action

    def _learn(self, **kwargs):
        """
        Update Q_actor using the Bellman Equation

        observations, actions, rewards should all be either
        normalized or standardized

        args
            observations        : np array (batch_size, observation_dim)
            actions             : np array (batch_size, num_actions)
            rewards             : np array (batch_size, 1)
            next_observations   : np array (batch_size, observataion_dim)
            episode             : int   

        returns
            history             : list
        """
        observations = kwargs.pop('observations')
        actions = kwargs.pop('actions')
        rewards = kwargs.pop('rewards')
        next_observations = kwargs.pop('next_observations')
        episode = kwargs.pop('episode')

        #  check that we have equal number of all of our inputs
        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == rewards.shape[0]
        assert observations.shape[0] == next_observations.shape[0]

        #  iterate over the experience to create the input and target 
        inputs = np.zeros(shape=(observations.shape[0],
                                 self.observation_dim + self.num_actions))
        targets = []
        print('starting input & target creation')
        for j, (obs, act, rew, next_obs) in enumerate(zip(observations,
                                                          actions, 
                                                          rewards,
                                                          next_observations)):
            #  first the inputs
            inputs[j] = np.append(obs, act)

            #  second the targets
            if next_obs.all() == -999999:
                #  if the current state is terminal
                #  the return is equal to the reward
                #  i.e. Q(s',a) = 0 for all a
                target = rew
            else:
                #  for non terminal states
                #  get all possible combinations of our next state
                #  across the action space
                state_actions, _ = self.all_state_actions(self.action_space,
                                                          next_obs)

                #  now predict the value of each of the state_actions
                #  note that we use Q_target here
                all_preds = []
                for sa in state_actions:
                    sa = sa.reshape(-1, state_actions.shape[1])
                    pred = self.Q_target.predict(sa)
                    all_preds.append(pred)

                all_preds = np.array(all_preds).reshape(-1)

                #  we take the maximum value across all actions
                max_Q = np.max(all_preds)

                #  the Bellman equation
                target = rew + self.discount * max_Q

            targets.append(target)

        targets = np.array(targets)

        #  update our Q function
        print('improving Q_actor')
        hist = self.Q_actor.improve(state_actions=inputs,
                                       targets=targets)

        self.memory.losses.append(hist.history['loss'][-1])

        #  copy weights over to target model
        if episode % 5 == 0:
            self.Q_target.copy_weights(parent=self.Q_actor.model)
        return hist
