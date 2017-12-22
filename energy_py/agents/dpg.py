

from energy_py.agents

class DPG(BaseAgent):
    """
    energy_py implementation of Determinstic Policy Gradients

    args
        env (object) energy_py environment
        actor (object) a determinstic policy
        critic (object) 


    references
        Silver et. al (2014) Determinstic Policy Gradient Algorithms
        http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

    """

    def __init__(self,
                 env,
                 discount,
                 actor, 
                 critic,
                 actor_noise,
                 batch_size,
                 brain_path=[],
                 ):

        super().__init__(env, discount, brain_path, memory_length)

        self.model_dict_actor = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': self.action_space.shape[0],
                           'layers': [25, 25],
                           'lr': 0.0025,
                           'tau': 0.001,
                           'batch_size': batch_size,
                           'epochs': 1}

        self.model_dict_critic = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': 1,
                           'layers': [25, 25],
                           'lr': 0.0025,
                           'tau': 0.001,
                           'batch_size': batch_size,
                           'epochs': 1}

        self.state_processor = Standardizer(self.observation_space.shape[0])
        self.action_processor = Standardizer(self.action_space.shape[0])
        self.target_processor = Normalizer(1) 

        #  create the actor & critic
        self.actor = actor(model_dict)
        self.critic = critic(model_dict)

        #  create actor noise with same dimension as action
        self.actor_noise = actor_noise(self.action_space.shape[0])

    def _reset(self):
        raise NotImplementedError

    def _act(self, **kwargs):

        sess = kwargs['sess']
        obs = kwargs['obs']

        obs = self.state_processor.transform(obs)

        determ_action = self.actor.get_action(sess, obs) 
        actor_noise = self.actor_noise()
        action = determ_action + actor_noise 
        action = np.array(action).reshape(1, self.action_space.shape[0])

        self.memory.info['determinsitic_action'] = determ_action
        self.memory.info['action_noise'] = actor_noise
        self.memory.info['action'] = action

        return action

    def _learn(self, **kwargs):

        sess = kwargs['sess']
        batch = kwargs['batch']

        #  unpack the batch dictionary
        obs = batch['obs']
        actions = batch['actions']
        rews = batch['rews']
        next_obs = batch['next_obs']
        terminal = batch['terminal']

        #  process our obs, actions and next_obs using our state processor
        obs = self.state_processor.transform(obs)
        actions = self.action_processor.transform(actions)
        next_obs = self.state_processor.transform(next_obs)

        #  create a Bellman target to update our critic 
        #  get an estimate of next state value using target network
        Q_next_obs = self.critic.predict_target(sess, obs, action)

        #  set terminal next state Q to zero
        Q_next_obs[terminal] = 0

        #  create the Bellman target
        targets = rews + self.discount * Q_next_obs

        #  save the unscaled targets 
        self.memory.info['unscaled_targets'].extend(targets.flatten().tolist())

        #  scale the targets
        targets = self.target_processor.transform(targets)

        #  run our training op
        error, loss = self.improve(sess, obs, action, targets)

        #  save data for analysis later
        self.memory['train_error'].extend(error.flatten().tolist())
        self.memory['loss'].append(loss)

        self.memory['scaled_obs'].extend(obs.flatten().tolist())
        self.memory.info['scaled_targets'].extend(targets.flatten().tolist())

        self.memory.info['max_target'].append(np.max(targets))
        self.memory.info['avg_target'].append(np.mean(targets))

        return {'error': error, 'loss': loss}


