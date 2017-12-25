import logging

import numpy as np

from energy_py import Normalizer, Standardizer
from energy_py.agents import BaseAgent


logger = logging.getLogger(__name__)

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
                 critic):

        super().__init__(env, discount)

        model_dict_actor = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': self.action_space.shape[0],
                           'layers': [25, 25],
                           'lr': 0.0025,
                           'tau': 0.001,
                           'action_space': self.action_space}

        model_dict_critic = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': 1,
                           'layers': [25, 25],
                           'lr': 0.0025,
                           'tau': 0.001,
                           'observation_space': self.observation_space,
                           'action_space': self.action_space}

        self.state_processor = Standardizer(self.observation_space.shape[0])
        self.action_processor = Standardizer(self.action_space.shape[0])
        self.target_processor = Normalizer(1) 

        #  create the actor & critic
        self.actor = actor(model_dict_actor)
        self.critic = critic(model_dict_critic, self.actor.num_vars)

    def _reset(self):
        raise NotImplementedError

    def _act(self, **kwargs):

        sess = kwargs['sess']
        obs = kwargs['obs']

        obs = self.state_processor.transform(obs)

        determ_action, noise, action = self.actor.get_action(sess, obs) 
        action = np.array(action).reshape(1, self.action_space.shape[0])

        self.memory.info['determinsitic_action'] = determ_action
        self.memory.info['noise'] = noise
        self.memory.info['action'] = action

        return action

    def _learn(self, **kwargs):

        sess = kwargs['sess']
        batch = kwargs['batch']

        #  unpack the batch dictionary
        obs = batch['obs']
        actions = batch['actions']
        rews = batch['rewards']
        next_obs = batch['next_obs']
        terminal = batch['terminal']

        #  process our obs, actions and next_obs using our state processor
        obs = self.state_processor.transform(obs)
        next_obs = self.state_processor.transform(next_obs)

        #  create a Bellman target to update our critic 
        #  get an estimate of next state value using target network
        #  we use the target network to generate actions
        t_actions = self.actor.get_target_action(sess, obs)
        t_actions = self.action_processor.transform(t_actions)
        Q_next_obs = self.critic.predict_target(sess, obs, t_actions)

        #  set terminal next state Q to zero
        Q_next_obs[terminal] = 0

        #  create the Bellman target
        targets = rews + self.discount * Q_next_obs

        #  save the unscaled targets 
        self.memory.info['unscaled_targets'].extend(targets.flatten().tolist())

        #  scale the targets
        targets = self.target_processor.transform(targets)

        #  update the critic 
        actual_actions = self.action_processor.transform(actions)
        c_error, c_loss = self.critic.improve(sess, obs, actual_actions, targets)

        #  save data for analysis later
        self.memory.info['c_train_error'].extend(c_error.flatten().tolist())
        self.memory.info['c_loss'].append(c_loss)

        self.memory.info['scaled_obs'].extend(obs.flatten().tolist())
        self.memory.info['scaled_targets'].extend(targets.flatten().tolist())

        self.memory.info['max_target'].append(np.max(targets))
        self.memory.info['avg_target'].append(np.mean(targets))

        #  update the actor
        #  get the actions the actor would take for these observations
        #  using online network (not target)
        actor_actions, _, _ = self.actor.get_action(sess, obs)

        #  get the gradients for these actions
        act_grads = self.critic.get_action_grads(sess, obs, actor_actions)[0]
        a_loss = self.actor.improve(sess, obs, act_grads)
        self.memory.info['a_loss'].append(a_loss)

        return {'c_error': c_error, 'c_loss': c_loss, 'a_loss:': a_loss}


