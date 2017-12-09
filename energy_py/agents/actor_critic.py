import logging

from energy_py.agents import BaseAgent

logger = logging.getLogger(__name__)


class ActorCritic(BaseAgent):
    """
    Parameterize two functions
        actor using an energy_py policy approximator
        critic using an energy_py value function approximator

    The critic sends the temporal difference error for the experienced s and s'
    to the actor, which updates policy parameters using the score function.

    args
        env (object) energy_py environment
        discount (float)
        brain_path (str)
        policy (object) energy_py policy approximator
        value_function (object) energy_py value function approximator
    """
    def __init__(self,
                 env,
                 discount,
                 brain_path,
                 policy,
                 value_function):

        #  initalizing the BaseAgent parent  
        super().__init__(env, discount, brain_path)

	#  create the actor
        #  a policy that maps state to action 
        actor_dict = {'input_nodes': self.observation_space.shape[0],
                      'output_nodes': self.action_space.shape[0]*2,
                      'layers': [25, 25],
                      'lr': 0.0001,
                      'action_space': self.action_space}
        self.actor = policy(actor_dict)

        #  create our critic
        #  on-policy estimate of the expected return for the actor 
        self.critic_dict = {'input_nodes': self.observation_space.shape[0],
                            'output_nodes': 1,
                            'layers': [25, 25],
                            'lr': 0.0025,
                            'batch_size': batch_size}
        self.critic = value_function(critic_model_dict)

        #  we make a state processor using the observation space
        #  minimums and maximums
        self.state_processor = Standardizer(self.observation_space.shape[0])

        #  we use a normalizer for the returns as well
        #  because we don't want to shift the mean
        self.returns_processor = Normalizer(1)

    def _act(self, **kwargs):
        """
        Act according to the policy network

        args
            observation : np array (1, observation_dim)
            session     : a TensorFlow Session object

        return
            action      : np array (1, num_actions)
        """
        observation = kwargs.pop('observation')
        session = kwargs.pop('session')

        #  scaling the observation for use in the policy network
        scaled_observation = self.state_processor.transform(observation.reshape(1,-1))

        #  generating an action from the policy network
        action, output = self.policy.get_action(session, scaled_observation)

        return action.reshape(1, self.num_actions)

    def _learn(self, **kwargs):
        """
        Update the critic and then the actor

        The critic uses the temporal difference error to update the actor

        args
            batch (np.array) a batch of experience to learn from
        return
            loss         : np float
        """
        sess = kwargs.pop('session')
        batch = kwargs.pop('batch')

        #  first we update the critic
        #  create a target using a Bellman estimate
        rews = batch[:, 2].reshape(-1, 1) 

        next_obs = np.concatenate(batch[:, 3])
        next_obs = next_obs.reshape(-1, self.observation_space.shape[0])
        next_obs = self.state_processor(next_obs)

        target = rews + self.discount * self.critic.predict(session, next_obs)
        target = target.reshape(-1, 1)
        self.memory.info['value fctn target'].append(target)

        #  then we improve the critic using the target
        obs = np.concatenate(batch[:, 0])
        obs = obs.reshape(-1, self.observation_space.shape[0])
        obs = self.state_processor(obs)

        #  we hve to send an action index into the energy_py value function
        #  update.  as this is approximating V(s), we index at 0
        act_index = np.zeros(self.obs.shape[0]).reshape(-1, 1)
        td_error, critic_loss = self.critic.improve(session, obs, target,
                                                    action_index=act_index)

        #  now we can update the actor
        #  we use the temporal difference error from the critic 
        actor_loss = self.actor.improve(session,
                                         obs,
                                         actions,
                                         td_error)

        #  output dict to iterate over for saving and printing
        output = {'td_error': td_error,
                  'critic_loss': critic_loss,
                  'actor_loss': actor_loss}

        for k, v in output.items():
            self.memory.info[k].append(v)
            logger.info('{} is {:.4f}'.format(k, v))

        #  only calc this so we can return something
        #  makes me think I should do one train op on this loss rather than
        #  two train_ops (critic+actor)
        loss = critic_loss + actor_loss

        return loss
