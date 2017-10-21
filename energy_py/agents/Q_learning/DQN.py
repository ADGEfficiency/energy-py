from energy_py.agents import Base_Agent

class Q_Learner(Base_Agent):
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
        batch_size          : int

    inherits from
        Base_Agent          : the energy_py class used for agents

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self, env,
                       Q_actor,
                       Q_target,
                       epsilon_decay_steps=10000,
                       batch_size=64):

    #  passing the environment to the Base_Agent class
    super().__init__(env, epsilon_decay_steps)

    #  create an object to decay epsilon
    self.e_greedy = Epsilon_Greedy(decay_steps=epsilon_decay_steps,
                                   verbose=0)

    def _act(self, **kwargs):
        """
        Act using an epsilon-greedy policy

        args
            observation : np array (1, observation_dim)
            Q_actor     : energy_py action-value function

        return
            action      : np array (1, num_actions)
        """
        #  pull out the relevant kwargs
        observation = kwargs.pop('observation')
        Q_actor = kwargs.pop('Q_actor')

        #  because our observation comes directly from the env
        #  we need to scale the observation
        observation = self.scale_array(observation, self.observation_space)

        #  get the current value of epsilon
        epsilon = self.e_greedy.get_epsilon()
        self.verbose_print('epsilon is {:.3f}'.format(epsilon))

        if epsilon > np.random.uniform():

            self.verbose_print('acting randomly')
            action = [space.sample() for space in self.action_space]

        else:
            self.verbose_print('acting according to Q_actor')

            #  create all possible combinations of our single observation
            #  and our n-dimensional action space
            state_acts, acts = self.all_state_acts(self.action_space,
                                                   observation)

            #  get predictions from the action_value function Q
            Q_estimates = [Q_actor.predict(sa) for sa in state_acts]

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
            observations        : np array (episode_length, observation_dim)
            actions             : np array (episode_length, num_actions)
            rewards             : np array (episode_length, 1)
            Q_target            : energy_py action-value function
            Q_actor             : energy_py action-value function

        returns
            Q_target            : energy_py action-value function
            Q_actor             : energy_py action-value function
            history             : list
        """
        observations = kwargs.pop('observations')
        actions = kwargs.pop('actions')
        rewards = kwargs.pop('rewards')
        next_observations = kwargs.pop('next_observations')

        Q_actor = kwargs.pop('Q_actor')
        Q_target = kwargs.pop('Q_target')

        #  check that we have equal number of all of our inputs
        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == rewards.shape[0]
        assert observations.shape[0] == next_observations.shape[0]

        #  first we create our targets
        targets = []
        for rew, next_obs in zip(rewards, next_observations):
            if next_obs == 'terminal':
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
                all_preds = np.array([Q_target.predict(sa) for
                                      sa in state_actions]).reshape(-1)

                #  we take the maximum value across all actions
                max_Q = np.max(all_preds)

                #  the Bellman equation
                target = rew + self.discount * max_Q

            #  add to our list of targets
            targets.append(target)

        #  reshape our targets into shape (num observations, 1)
        targets = np.array(targets).reshape(-1, 1)
        assert targets.shape[0] == observations.shape[0]

        #  create the input for our Q network
        inputs, _ = self.all_state_actions(self.action_space, obs)

        #  update our Q function
        history = Q_actor.improve(x=inputs, y=targets)

        return Q_target, Q_actor, history
