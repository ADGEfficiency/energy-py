import os

import numpy as np

from energy_py.agents import Base_Agent, Epsilon_Greedy


class DQN(Base_Agent):
    """
    energy_py implementation of DQN
    aka Q-learning with experience replay & target network

    args
        env                 : energy_py environment
        Q                   : energy_py Action-Value Function Q(s,a)
        discount
        batch_size
        epsilon_decay_steps : int
        update_target_net   : int : steps before target network update
        memory_length       : int : length of experience replay
        scale_targets       : bool : whether to scale Q(s,a) when learning

    inherits from
        Base_Agent          : the energy_py class used for agents

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self, env,
                       Q,
                       discount,
                       batch_size,
                       epsilon_decay_steps=10000,
                       epsilon_start=1.0,
                       update_target_net=100,
                       memory_length=100000,
                       scale_targets=True,
                       brain_path=[],
                       load_agent_brain=True,
                       verbose=False):

        #  passing the environment to the Base_Agent class
        super().__init__(env, discount, brain_path,
                         memory_length, verbose)

        self.epsilon_decay_steps = epsilon_decay_steps
        self.update_target_net = update_target_net
        self.scale_targets = scale_targets

        #  model dict gets passed into the Action-Value function objects
        model_dict = {'type' : 'feedforward',
                      'input_dim' : self.observation_dim + self.num_actions,
                      'layers'    : [25],
                      'output_dim': 1,
                      'lr'        : 0.001,
                      'batch_size': 32,
                      'epochs'    : 1}

        #  make our two action value functions
        self.Q_actor = Q(model_dict)
        self.Q_target = Q(model_dict)

        #  create an object to decay epsilon
        self.e_greedy = Epsilon_Greedy(decay_steps=epsilon_decay_steps,
                                       epsilon_start=epsilon_start)

        if load_agent_brain:
            self.load_brain()

    def _reset(self):
        """
        Resets the agent
        """
        self.Q_actor.model.reset_weights()
        self.Q_target.model.reset_weights()
        self.e_greedy.reset()

    def _act(self, observation):
        """
        Act using an epsilon-greedy policy

        args
            observation : np array (1, observation_dim)

        return
            action      : np array (1, num_actions)
        """
        #  because our observation comes directly from the env
        #  we need to scale the observation
        observation = self.scale_array(observation, self.observation_space)

        #  get the current value of epsilon
        epsilon = self.e_greedy.epsilon
        self.verbose_print('epsilon is {:.3f}'.format(epsilon), level=2)
        self.memory.agent_stats['epsilon'].append(epsilon)

        if np.random.uniform() < epsilon:
            self.verbose_print('acting randomly', level=2)
            action = [space.sample() for space in self.action_space]

        else:

            #  create all possible combinations of our single observation
            #  and our n-dimensional action space
            state_acts, acts = self.all_state_actions(self.action_space,
                                                      observation)

            #  get predictions from the action_value function Q
            Q_estimates = [self.Q_actor.predict(sa.reshape(1,-1))
                           for sa in state_acts]

            #  select the action with the highest Q
            #  note that we index the unscaled action
            #  as this action is sent directly to the environment
            action = acts[np.argmax(Q_estimates)]
            self.verbose_print('acting according to Q_actor - max Q(s,a)={}'.format(np.max(Q_estimates), level=2))

            #  save the Q estimates
            self.memory.agent_stats['acting max Q estimates'].append(np.max(Q_estimates))

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
        self.verbose_print('starting input & target creation', level=2)
        for j, (obs, act, rew, next_obs) in enumerate(zip(observations,
                                                          actions,
                                                          rewards,
                                                          next_observations)):
            #  first the inputs
            inputs[j] = np.append(obs, act)

            #  second the targets
            if next_obs.all() == -999999:
                #  if the next state is terminal
                #  the return of our current state is equal to the reward
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
                max_q = max([self.Q_target.predict(sa.reshape(-1, state_actions.shape[1]))
                             for sa in state_actions])

                #  the Bellman equation
                target = rew + self.discount * max_q

            targets.append(target)

        #  now targets are all done, turn our list into a numpy array
        #  this is so we can scale using normalization
        targets = np.array(targets)
        mean_unscaled_targets = np.mean(targets)

        #  scaling the targets by normalizing
        if self.scale_targets:
            targets = (targets - targets.min()) / (targets.max() - targets.min())

        #  reshape targets into 2 dimensions
        targets = targets.reshape(-1,1)

        #  update our Q function
        self.verbose_print('Improving Q_actor - avg unscaled target={}'.format(mean_unscaled_targets), level=1)
        self.verbose_print('input shape {}'.format(inputs.shape), level=2)
        self.verbose_print('target shape {}'.format(targets.shape), level=2)

        hist = self.Q_actor.improve(state_actions=inputs,
                                    targets=targets)

        self.memory.agent_stats['loss'].append(hist.history['loss'][-1])
        self.memory.agent_stats['training Q targets'].extend(targets.tolist())

        return hist

    def update_target_network(self):

        #  copy weights over to target model
        self.verbose_print('updating target network', level=0)
        self.Q_target.copy_weights(parent=self.Q_actor.model)


    def _load_brain(self):
        """
        Loads memory, Q_actor and Q_target

        TODO repeated code, maybe put this into Base_Agent init
        """
        brain = ['experiences.pickle', 'Q_actor.h5', 'Q_target.h5']
        paths = {key:os.path.join(self.brain_path, key) for key in brain}

        #  load the experiences into the Agent_Memory object
        experiences = self.load_pickle(paths['experiences.pickle'])
        self.memory.add_experience_list(experiences)

        #  load the action value functions
        self.Q_actor.load_model(paths['Q_actor.h5'])
        self.Q_target.load_model(paths['Q_target.h5'])

    def _save_brain(self):
        """
        Saves experiences, Q_actor and Q_target
        """
        brain = ['experiences.pickle', 'Q_actor.h5', 'Q_target.h5']
        paths = {key:os.path.join(self.brain_path, key) for key in brain}
        [self.ensure_dir(path) for key, path in paths.items()]

        #  save the experience list
        self.dump_pickle(self.memory.experiences, paths['experiences.pickle'])

        #  not reccomended to use pickle for Keras models
        #  so we use h5py to save Keras models
        self.Q_actor.save_model(paths['Q_actor.h5'])
        self.Q_target.save_model(paths['Q_target.h5'])

