#from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

import numpy as np
import base_net_ddpg as base_network
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util

class CriticNetwork(base_network.Network):
  """ the critic represents a mapping from state & actors action to a quality score."""

  def __init__(self, namespace, actor, opts):
    super(CriticNetwork, self).__init__(namespace)

    # input state to the critic is the _same_ state given to the actor.
    # input action to the critic is simply the output action of the actor.
    # even though when training we explicitly provide a new value for the
    # input action (via the input_action placeholder) we need to be stop the gradient
    # flowing to the actor since there is a path through the actor to the input_state
    # too, hence we need to be explicit about cutting it (otherwise training the
    # critic will attempt to train the actor too.
    self.input_state = actor.input_state
    self.internal_state = actor.internal_state
    self.target_obj = actor.target_obj
    self.opts = opts
    self.input_action = tf.stop_gradient(actor.output_action)

    with tf.variable_scope(namespace):
      conv_net = self.simple_conv_net_on(self.input_state, self.opts)
      flat_conv_net = slim.flatten(conv_net, scope='flat')
      final_hidden = self.simple_critic_fc_net_on(flat_conv_net, self.input_action, self.internal_state, self.target_obj, self.opts)
      self.q_value = slim.fully_connected(scope='q_value',
                                          inputs=final_hidden,
                                          num_outputs=1,
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                          activation_fn=None)

#      print(self.q_value.name)

  def init_ops_for_training(self, target_critic):
  # update critic using bellman equation; Q(s1, a) = reward + discount * Q(s2, A(s2))

    # left hand side of bellman is just q_value, but let's be explicit about it...
    bellman_lhs = self.q_value

    # right hand side is ...
    #  = reward + discounted q value from target actor & critic in the non terminal case
    #  = reward  # in the terminal case
    self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="critic_reward")
    self.terminal_mask = tf.placeholder(shape=[None, 1], dtype=tf.float32,
                                        name="critic_terminal_mask")
    self.input_state_2 = target_critic.input_state
    bellman_rhs = self.reward + (self.terminal_mask * self.opts.discount * target_critic.q_value)

    # note: since we are NOT training target networks we stop gradients flowing to them
    bellman_rhs = tf.stop_gradient(bellman_rhs)

    # the value we are trying to mimimise is the difference between these two; the
    # temporal difference we use a squared loss for optimisation and, as for actor, we
    # wrap optimiser in a namespace so it's not picked up by target network variable
    # handling.
    self.temporal_difference = bellman_lhs - bellman_rhs
    self.temporal_difference_loss = tf.reduce_mean(tf.pow(self.temporal_difference, 2))
#    self.temporal_difference_loss = tf.Print(self.temporal_difference_loss, [self.temporal_difference_loss], 'temporal_difference_loss')
#    with tf.variable_scope(self.namespace):
    with tf.variable_scope("optimiser_critic"):
      # calc gradients
      optimiser = tf.train.GradientDescentOptimizer(self.opts.critic_learning_rate)
      gradients = optimiser.compute_gradients(self.temporal_difference_loss)
      # potentially clip and wrap with debugging tf.Print
      gradients = util.clip_and_debug_gradients(gradients, self.opts)
      # apply
      self.train_op = optimiser.apply_gradients(gradients)

      print(self.train_op)

  def q_gradients_wrt_actions(self):
    """ gradients for the q.value w.r.t just input_action; used for actor training"""
    return tf.gradients(self.q_value, self.input_action)[0]

#  def debug_q_value_for(self, input_state, action=None):
#    feed_dict = {self.input_state: input_state}
#    if action is not None:
#      feed_dict[self.input_action] = action
#    return np.squeeze(tf.get_default_session().run(self.q_value, feed_dict=feed_dict))

  def train(self, batch):
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state: batch.state_1,
                                            self.input_action: batch.action,
                                            self.reward: batch.reward,
                                            self.terminal_mask: batch.terminal_mask,
                                            self.input_state_2: batch.state_2,
                                            self.internal_state: batch.internal_state,
                                            self.target_obj: batch.target_obj,
                                            self.IS_TRAINING: True})

  def check_loss(self, batch):
    return tf.get_default_session().run([self.temporal_difference_loss,
                                         self.temporal_difference,
                                         self.q_value],
                                        feed_dict={self.input_state: batch.state_1,
                                                   self.input_action: batch.action,
                                                   self.reward: batch.reward,
                                                   self.terminal_mask: batch.terminal_mask,
                                                   self.input_state_2: batch.state_2,
                                                   self.internal_state: batch.internal_state,
                                                   self.target_obj: batch.target_obj,
                                                   self.IS_TRAINING: False})