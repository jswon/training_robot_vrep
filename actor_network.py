import numpy as np
import base_net_ddpg as base_network
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util

VERBOSE_DEBUG = False

class ActorNetwork(base_network.Network):
  """ the actor represents the learnt policy mapping states to actions"""

  def __init__(self, namespace, input_state, internal_state, target_obj, action_dim, opts):
    super(ActorNetwork, self).__init__(namespace)

    self.input_state = input_state
    self.internal_state = internal_state
    self.target_obj = target_obj
    self.opts = opts

    self.exploration_noise = util.OrnsteinUhlenbeckNoise(action_dim,
                                                         opts.action_noise_theta,
                                                         opts.action_noise_sigma)

    with tf.variable_scope(namespace):
      conv_net = self.simple_conv_net_on(self.input_state, opts)
      flat_conv_net = slim.flatten(conv_net, scope='flat')
      self.output_action = self.simple_actor_fc_net_on(flat_conv_net, action_dim, internal_state, target_obj, opts)

#      print(self.output_action.name)

  def init_ops_for_training(self, critic):
    # actors gradients are the gradients for it's output w.r.t it's vars using initial
    # gradients provided by critic. this requires that critic was init'd with an
    # input_action = actor.output_action (which is natural anyway)
    # we wrap the optimiser in namespace since we don't want this as part of copy to
    # target networks.
    # note that we negate the gradients from critic since we are trying to maximise
    # the q values (not minimise like a loss)
#    with tf.variable_scope(self.namespace):
    with tf.variable_scope("optimiser_actor"):
      gradients = tf.gradients(self.output_action,
                               self.trainable_model_vars(),
                               tf.negative(critic.q_gradients_wrt_actions()))
      gradients = zip(gradients, self.trainable_model_vars())
      # potentially clip and wrap with debugging
      gradients = util.clip_and_debug_gradients(gradients, self.opts)
      # apply
      optimiser = tf.train.GradientDescentOptimizer(self.opts.actor_learning_rate)
      self.train_op = optimiser.apply_gradients(gradients)

      print(self.train_op.name)

  def action_given(self, state, internal_state, target_obj, add_noise=False):
    # feed explicitly provided state
    actions = tf.get_default_session().run(self.output_action,
                                           feed_dict={self.input_state: [state],
                                                      self.internal_state: [internal_state],
                                                      self.target_obj: [target_obj],
                                                      self.IS_TRAINING: False})

    # NOTE: noise is added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph required for online training. it's
    # only used during training after being the replay buffer.
    if add_noise:
      if VERBOSE_DEBUG:
        pre_noise = str(actions)
      actions[0] += self.exploration_noise.sample()

    actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)
    actions = actions * 360.0

    actions[0,0] = (actions[0,0] / 4.0) - 90
    actions[0,2] = actions[0,2] / 2.0

    actions[0,2] = actions[0,2] / 2.0

    return actions

  def train(self, state, internal_state, target_obj):
    # training actor only requires state since we are trying to maximise the
    # q_value according to the critic.
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state: state,
                                            self.internal_state: internal_state,
                                            self.target_obj: target_obj,
                                            self.IS_TRAINING: True})



