import logging
import os

import numpy as np

from energy_py import Normalizer, Standardizer
from energy_py.agents import BaseAgent, EpsilonGreedy

logger = logging.getLogger(__name__)


class DQN(BaseAgent):
    """
    energy_py implementation of DQN
    aka Q-learning with experience replay & target network

    args
        env                 : energy_py environment
        Q                   : energy_py Action-Value Function Q(s,a)
        discount
        batch_size

    inherits from
        Base_Agent          : the energy_py class used for agents

    Based on the DeepMind Atari work
    Reference = Mnih et. al (2013), Mnih et. al (2015)
    """
    def __init__(self,
                 env,
                 discount,
                 Q,
                 batch_size,
                 total_steps,
                 brain_path=[],
                 load_agent_brain=False):
        """
        DeepMind Atari hyperparameters are here as an FYI

        batch_size = 32
        replay_memory_size = 1 million
        target network update freq = 10000
        replay start size = 50000 (steps of exp used to populate memory)
        learning rate = 0.00025
        total frames = 10 million
        """
        memory_length = int(total_steps * 0.1)
        self.epsilon_decay_steps = int(total_steps / 2)
        self.update_target_net = int(total_steps * 0.0125)
        self.initial_random = int(total_steps * 0.1)

        #  passing the environment to the BaseAgent class
        super().__init__(env, discount, brain_path, memory_length)

        #  Q-Learning needs a discrete action space
        #
        self.actions = self.action_space.discretize(10)

        self.model_dict = {'input_nodes': self.observation_space.shape[0],
                      'output_nodes': self.actions.shape[0],
                      'layers': [25, 25],
                      'lr': 0.0025,
                      'batch_size': batch_size,
                      'epochs': 1}

        self.state_processor = Standardizer(self.observation_space.shape[0])
        self.target_processor = Normalizer(1)

        #  make our two action value functions
        self.Q_actor = Q(self.model_dict, 'Q_actor')
        self.Q_target = Q(self.model_dict, 'Q_target')

        #  create an object to decay epsilon
        self.e_greedy = EpsilonGreedy(self.initial_random,
                                      self.epsilon_decay_steps)

        if load_agent_brain:
            self.load_brain()

    def _reset(self):
        """
        Resets the agent
        """
        self.Q_actor.model.reset_weights()
        self.Q_target.model.reset_weights()
        self.e_greedy.reset()

    def _act(self, **kwargs):
        """
        Act using an epsilon-greedy policy

        args
            sess (tf.Session): the current tensorflow session
            obs (np.array): shape=(1, observation_space.shape[0])

        return
            action      : np array (1, action_space.shape[0])
        """
        sess = kwargs.pop('sess')
        observation = kwargs.pop('obs')

        #  using our state processor to transform the observation
        observation = self.state_processor.transform(observation)

        #  get the current value of epsilon
        epsilon = self.e_greedy.epsilon
        self.memory.info['epsilon'].append(epsilon)

        if np.random.uniform() < epsilon:
            #  acting randomly
            logger.debug('epsilon {:.3f} - acting randomly'.format(epsilon))
            action = self.action_space.sample_discrete()

        else:
            #  get our Q(s,a) approximation for each action
            Q_est = self.Q_actor.predict(sess, observation)

            #  select the action with the highest Q
            action = self.actions[np.argmax(Q_est)]

            #  calculate some statistics from the Q estimates
            max_Q = np.max(Q_est)
            avg_Q = np.mean(Q_est)

            #  save some data for debugging later
            self.memory.info['max_Q_acting estimates'].append(max_Q)
            self.memory.info['avg_Q_acting estimates'].append(avg_Q)
            self.memory.info['Q act est'].extend(Q_est.flatten().tolist())

            logger.debug('using Q_actor - max(Q_est)={:.3f}'.format(max_Q))
            logger.debug('using Q_actor - avg(Q_est)={:.3f}'.format(avg_Q))

        #  make sure action is shaped correctly
        action = np.array(action).reshape(1, self.action_space.shape[0])
        assert self.action_space.shape[0] == action.shape[1]

        return action

    def _learn(self, **kwargs):
        """
        Update Q_actor using the Bellman Equation

        observations, actions, rewards should all be either
        normalized or standardized

        Q(s', a) is calculated externally to the value function
        Q(s,a) is calculated within the value function

        args
            sess (tf.Session)
            batch (np.array): batch of experience to learn from

        returns
            train_info (dict)
        """
        sess = kwargs.pop('sess')
        batch = kwargs.pop('batch')

        obs = np.concatenate(batch[:, 0])
        obs = obs.reshape(-1, self.observation_space.shape[0])

        actions = np.concatenate(batch[:, 1])
        actions = actions.reshape(-1, self.action_space.shape[0]) 

        rews = batch[:, 2].reshape(-1, 1)

        next_obs = np.concatenate(batch[:, 3])
        next_obs = next_obs.reshape(-1, self.observation_space.shape[0])

        terminal = np.array(batch[:, 4],dtype=np.bool).reshape(-1)

        logger.debug('shapes of arrays used in learning')
        logger.debug('obs shape {}'.format(obs.shape))
        logger.debug('actions shape {}'.format(actions.shape))
        logger.debug('rews shape {}'.format(rews.shape))
        logger.debug('terminal shape {}'.format(terminal.shape))

        #  dimensionality checks
        assert rews.shape[0] == obs.shape[0]
        assert rews.shape[0] == actions.shape[0]
        assert rews.shape[0] == next_obs.shape[0]
        assert rews.shape[0] == terminal.shape[0]

        #  we process the entire batch of inputs using our state_processor
        inputs = self.state_processor.transform(obs)
        next_obs = self.state_processor.transform(next_obs)
        preds = self.Q_target.predict(sess, next_obs)
        #  change to max q next state
        max_q = np.max(self.Q_target.predict(sess, next_obs), axis=1)
        max_q = max_q.reshape(-1, 1)

        assert max_q.shape[0] == inputs.shape[0]

        #  we set the max Q(s',a) equal to zero for terminal states
        logger.debug('avg max_q before terminal {}'.format(np.mean(max_q)))
        max_q[terminal] = 0
        logger.debug('avg max_q after terminal {}'.format(np.mean(max_q)))

        #  we then use the Bellman equation with our masked max_q
        logger.debug('before bellman eqn')
        logger.debug('rews shape {}'.format(rews.shape))
        logger.debug('max_q shape {}'.format(max_q.shape))
        targets = rews + self.discount * max_q
        logger.debug('targets shape {}'.format(targets.shape))
        assert targets.shape[0] == inputs.shape[0]
        assert targets.shape[1] == 1

        #  save the unscaled targets so we can visualize later
        self.memory.info['unscaled Q targets'].extend(list(targets.flatten()))

        #  scaling the targets by normalizing
        targets = self.target_processor.transform(targets)
        assert targets.shape[0] == inputs.shape[0]

        #  action_index is flattened!!!
        act_list = self.actions.tolist()
        action_index = [act_list.index(act) for act in actions.tolist()]
        action_index = np.array(action_index).reshape(-1)

        logger.debug('action indicies shape {}'.format(action_index.shape))
        assert action_index.shape[0] == obs.shape[0]

        #  flattening!!!
        targets = targets.flatten()
        assert targets.shape[0] == inputs.shape[0]
        error, loss = self.Q_actor.improve(sess, inputs, targets, action_index)

        logger.debug('inputs shape {}'.format(inputs.shape))
        logger.debug('Q preds for next_obs shape {}'.format(preds.shape))
        logger.debug('max q values has shape {}'.format(max_q.shape))
        logger.debug('target shape after bellman equn {}'.format(targets.shape))
        logger.debug('target shape processing {}'.format(targets.shape))

        #  save loss and the training targets for visualization later
        self.memory.info['train error'].extend(error.flatten().tolist())
        self.memory.info['loss'].append(loss)

        self.memory.info['train Q inputs'].extend(inputs.flatten().tolist())
        self.memory.info['train Q targets'].extend(targets.flatten().tolist())

        #  save some data for analysis later
        max_target = np.max(targets)
        avg_target = np.mean(targets)

        self.memory.info['max learning target'].append(max_target)
        self.memory.info['avg learning target'].append(avg_target)
        self.memory.info['learning targets'].extend(targets.flatten().tolist())

        logger.debug('learning - max(Q_est)={:.3f}'.format(max_target))
        logger.debug('learning - avg(Q_est)={:.3f}'.format(avg_target))

        train_info = {'error': error,
                      'loss': loss}

        return train_info

    def _load_brain(self):
        """
        Loads experiences, Q_actor and Q_target
        """
        #  load the epsilon greedy object
        e_greedy_path = os.path.join(self.brain_path, 'e_greedy.pickle')
        self.e_greedy = self.load_pickle(e_greedy_path)

        #  load the action value functions
        Q_actor_path = os.path.join(self.brain_path, 'Q_actor.h5')
        self.Q_actor.load_model(Q_actor_path)
        self.Q_target = self.Q_actor

    def _save_brain(self):
        """
        Saves experiences, Q_actor and Q_target
        """
        e_greedy_path = os.path.join(self.brain_path, 'e_greedy.pickle')
        self.dump_pickle(self.e_greedy, e_greedy_path)

        """
        add the acting Q network
        we don't add the target network - we just use the acting network
        to initialize Q_target when we load_brain
        not reccomended to use pickle for Keras models
        so we use h5py to save Keras models
        """
        Q_actor_path = os.path.join(self.brain_path, 'Q_actor.h5')
        self.Q_actor.save_model(Q_actor_path)

    def update_target_network(self, sess):
        """
        Copies weights from Q_actor into Q_target
        """
        logger.info('Updating Q_target by copying weights from Q_actor')
        sess = self.Q_target.copy_weights(sess, parent=self.Q_actor)
        return sess
