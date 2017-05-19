#!/usr/bin/env python
import argparse
import collections
import datetime
import json
import numpy as np
import replay_memory
import signal
import sys
import random
import tensorflow as tf
import time
import util
import vrep_python_mod
import base_net_raw as base_network
import tensorflow.contrib.slim as slim
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-eval', type=int, default=0,
                    help="if >0 just run this many episodes with no training")
parser.add_argument('--max-num-actions', type=int, default=0,
                    help="train for (at least) this number of actions (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--max-run-time', type=int, default=0,
                    help="train for (at least) this number of seconds (always finish current episode)"
                         " ignore if <=0")
parser.add_argument('--ckpt-dir', type=str, default='ckpt/ddpg', help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=3600, help="freq (sec) to save ckpts")
parser.add_argument('--batch-size', type=int, default=32, help="training batch size")
parser.add_argument('--batches-per-step', type=int, default=5,
                    help="number of batches to train per step")
parser.add_argument('--dont-do-rollouts', action="store_true",
                    help="by dft we do rollouts to generate data then train after each rollout. if this flag is set we"
                         " dont do any rollouts. this only makes sense to do if --event-log-in set.")
parser.add_argument('--target-update-rate', type=float, default=0.0001,
                    help="affine combo for updating target networks each time we run a training batch")
parser.add_argument('--use-batch-norm', default=False, action="store_true",
                    help="whether to use batch norm on conv layers")
parser.add_argument('--actor-hidden-layers', type=str, default="200,200,50", help="actor hidden layer sizes")
parser.add_argument('--critic-hidden-layers', type=str, default="200,200,50", help="critic hidden layer sizes")
parser.add_argument('--actor-learning-rate', type=float, default=0.0001, help="learning rate for actor")
parser.add_argument('--critic-learning-rate', type=float, default=0.001, help="learning rate for critic")
parser.add_argument('--discount', type=float, default=0.99, help="discount for RHS of critic bellman equation update")
parser.add_argument('--event-log-in', type=str, default=None,
                    help="prepopulate replay memory with entries from this event log")
parser.add_argument('--replay-memory-size', type=int, default=10000, help="max size of replay memory")
parser.add_argument('--replay-memory-burn-in', type=int, default=100, help="dont train from replay memory until it reaches this size")
parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
parser.add_argument('--action-noise-theta', type=float, default=0.1,
                    help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.01,
                    help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
parser.add_argument('--joint-angle-low-limit', type=float, default=-180,
                    help="joint angle low limit for action")
parser.add_argument('--joint-angle-high-limit', type=float, default=180,
                    help="joint angle high limit for action")
parser.add_argument('--action_dim', type=float, default=6,
                    help="number of joint angle for robot action")
parser.add_argument('--internal_state_dim', type=float, default=18,
                    help="internal_state_dim")
parser.add_argument('--action_repeat_per_scene', type=float, default=20,
                    help="number of actions per a scene")
parser.add_argument('--number_of_scenes_per_shuffle', type=float, default=10,
                    help="number of scenes per a shuffle")
parser.add_argument('--use-full-internal-state', default=False, action="store_true",
                    help="whether to use full internal state")


util.add_opts(parser)

vrep_python_mod.add_opts(parser)
opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

# TODO: if we import slim _before_ building cartpole env we can't start bullet with GL gui o_O
env = vrep_python_mod.sim_env(opts=opts)

VERBOSE_DEBUG = False

def toggle_verbose_debug(signal, frame):
  global VERBOSE_DEBUG
  VERBOSE_DEBUG = not VERBOSE_DEBUG
  signal.signal(signal.SIGTERM, toggle_verbose_debug)
#signal.signal(signal.SIGUSR1, toggle_verbose_debug)

DUMP_WEIGHTS = False
def set_dump_weights(signal, frame):
  global DUMP_WEIGHTS
  DUMP_WEIGHTS = True
signal.signal(signal.SIGTERM, set_dump_weights)
#signal.signal(signal.SIGUSR2, set_dump_weights)


class ActorNetwork(base_network.Network):
  """ the actor represents the learnt policy mapping states to actions"""

  def __init__(self, namespace, input_state, action_dim):
    super(ActorNetwork, self).__init__(namespace)

    self.input_state = input_state

    self.exploration_noise = util.OrnsteinUhlenbeckNoise(action_dim,
                                                         opts.action_noise_theta,
                                                         opts.action_noise_sigma)

    with tf.variable_scope(namespace):
      opts.hidden_layers = opts.actor_hidden_layers
      final_hidden = self.input_state_network(self.input_state, opts)
      # action dim output. note: actors out is (-1, 1) and scaled in env as required.
      weights_initializer = tf.random_uniform_initializer(-0.001, 0.001)
      self.output_action = slim.fully_connected(scope='output_action',
                                                inputs=final_hidden,
                                                num_outputs=action_dim,
                                                weights_initializer=weights_initializer,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                activation_fn=tf.nn.tanh)

  def init_ops_for_training(self, critic):
    # actors gradients are the gradients for it's output w.r.t it's vars using initial
    # gradients provided by critic. this requires that critic was init'd with an
    # input_action = actor.output_action (which is natural anyway)
    # we wrap the optimiser in namespace since we don't want this as part of copy to
    # target networks.
    # note that we negate the gradients from critic since we are trying to maximise
    # the q values (not minimise like a loss)
    with tf.variable_scope("optimiser"):
      gradients = tf.gradients(self.output_action,
                               self.trainable_model_vars(),
                               tf.negative(critic.q_gradients_wrt_actions()))
      gradients = zip(gradients, self.trainable_model_vars())
      # potentially clip and wrap with debugging
      gradients = util.clip_and_debug_gradients(gradients, opts)
      # apply
      optimiser = tf.train.GradientDescentOptimizer(opts.actor_learning_rate)
      self.train_op = optimiser.apply_gradients(gradients)

  def action_given(self, state, add_noise=False):
    # feed explicitly provided state
    actions = tf.get_default_session().run(self.output_action,
                                           feed_dict={self.input_state: [state],
                                                      base_network.IS_TRAINING: False})

    # NOTE: noise is added _outside_ tf graph. we do this simply because the noisy output
    # is never used for any part of computation graph required for online training. it's
    # only used during training after being the replay buffer.
    if add_noise:
      if VERBOSE_DEBUG:
        pre_noise = str(actions)
      actions[0] += self.exploration_noise.sample()
      actions = np.clip(1, -1, actions)  # action output is _always_ (-1, 1)
      if VERBOSE_DEBUG:
        print("TRAIN action_given pre_noise %s post_noise %s" % (pre_noise, actions))

    return actions

  def train(self, state):
    # training actor only requires state since we are trying to maximise the
    # q_value according to the critic.
    tf.get_default_session().run(self.train_op,
                                 feed_dict={self.input_state: state,
                                            base_network.IS_TRAINING: True})


class CriticNetwork(base_network.Network):
  """ the critic represents a mapping from state & actors action to a quality score."""
  def __init__(self, namespace, actor):
    super(CriticNetwork, self).__init__(namespace)

    # input state to the critic is the _same_ state given to the actor.
    # input action to the critic is simply the output action of the actor.
    # even though when training we explicitly provide a new value for the
    # input action (via the input_action placeholder) we need to be stop the gradient
    # flowing to the actor since there is a path through the actor to the input_state
    # too, hence we need to be explicit about cutting it (otherwise training the
    # critic will attempt to train the actor too.
    self.input_state = actor.input_state
    self.input_action = tf.stop_gradient(actor.output_action)

    with tf.variable_scope(namespace):
      if opts.use_raw_pixels:
        conv_net = self.simple_conv_net_on(self.input_state, opts)
        # TODO: use base_network helper
        hidden1 = slim.fully_connected(conv_net, 200, scope='hidden1')
        hidden2 = slim.fully_connected(hidden1, 50, scope='hidden2')
        concat_inputs = tf.concat([hidden2, self.input_action], 1)
        final_hidden = slim.fully_connected(concat_inputs, 50, scope="hidden3")
      else:
        # stack of hidden layers on flattened input; (batch,2,2,7) -> (batch,28)
        flat_input_state = slim.flatten(self.input_state, scope='flat')
        concat_inputs = tf.concat([flat_input_state, self.input_action],1)
        final_hidden = self.hidden_layers_starting_at(concat_inputs,
                                                      opts.critic_hidden_layers)

      # output from critic is a single q-value
      self.q_value = slim.fully_connected(scope='q_value',
                                          inputs=final_hidden,
                                          num_outputs=1,
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                          activation_fn=None)

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
    bellman_rhs = self.reward + (self.terminal_mask * opts.discount * target_critic.q_value)

    # note: since we are NOT training target networks we stop gradients flowing to them
    bellman_rhs = tf.stop_gradient(bellman_rhs)

    # the value we are trying to mimimise is the difference between these two; the
    # temporal difference we use a squared loss for optimisation and, as for actor, we
    # wrap optimiser in a namespace so it's not picked up by target network variable
    # handling.
    self.temporal_difference = bellman_lhs - bellman_rhs
    self.temporal_difference_loss = tf.reduce_mean(tf.pow(self.temporal_difference, 2))
#    self.temporal_difference_loss = tf.Print(self.temporal_difference_loss, [self.temporal_difference_loss], 'temporal_difference_loss')
    with tf.variable_scope("optimiser"):
      # calc gradients
      optimiser = tf.train.GradientDescentOptimizer(opts.critic_learning_rate)
      gradients = optimiser.compute_gradients(self.temporal_difference_loss)
      # potentially clip and wrap with debugging tf.Print
      gradients = util.clip_and_debug_gradients(gradients, opts)
      # apply
      self.train_op = optimiser.apply_gradients(gradients)

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
                                            base_network.IS_TRAINING: True})

  def check_loss(self, batch):
    return tf.get_default_session().run([self.temporal_difference_loss,
                                         self.temporal_difference,
                                         self.q_value],
                                        feed_dict={self.input_state: batch.state_1,
                                                   self.input_action: batch.action,
                                                   self.reward: batch.reward,
                                                   self.terminal_mask: batch.terminal_mask,
                                                   self.input_state_2: batch.state_2,
                                                   base_network.IS_TRAINING: False})


class DeepDeterministicPolicyGradientAgent(object):
  def __init__(self, env):
    self.env = env
    self.obj_list = [i for i in range(10)]

    state_shape = self.env.state_shape
    action_dim = self.env.action_space.shape[1]

    # for now, with single machine synchronous training, use a replay memory for training.
    # this replay memory stores states in a Variable (ie potentially in gpu memory)
    # TODO: switch back to async training with multiple replicas (as in drivebot project)
    self.replay_memory = replay_memory.ReplayMemory(opts.replay_memory_size,
                                                    state_shape, action_dim, opts)

    # s1 and s2 placeholders
    #batched_state_shape = [None] + list(state_shape)
    batched_state_shape = [None] + [state_shape]
    s1 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
    s2 = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)

    if opts.use_raw_pixels :
      if opts.use_full_internal_state:
        temp = [18]
      else:
        temp = [9]

      batched_internal_state_shape = [None] + temp
      internal_state = tf.placeholder(shape=batched_internal_state_shape, dtype=tf.float32)
      temp = [10]

      batched_taget_obj_shape = [None] + temp
      taget_obj = tf.placeholder(shape=batched_taget_obj_shape, dtype=tf.float32)

    # initialise base models for actor / critic and their corresponding target networks
    # target_actor is never used for online sampling so doesn't need explore noise.
    self.actor = ActorNetwork("actor", s1, action_dim)
    self.critic = CriticNetwork("critic", self.actor)

    self.target_actor =ActorNetwork("target_actor", s2, action_dim)
    self.target_critic = CriticNetwork("target_critic", self.target_actor)

    # setup training ops;
    # training actor requires the critic (for getting gradients)
    # training critic requires target_critic (for RHS of bellman update)
    self.actor.init_ops_for_training(self.critic)
    self.critic.init_ops_for_training(self.target_critic)

  def post_var_init_setup(self):
    # prepopulate replay memory (if configured to do so)
    if opts.event_log_in:
      self.replay_memory.reset_from_event_log(opts.event_log_in)
    # hook networks up to their targets
    # ( does one off clobber to init all vars in target network )
    self.target_actor.set_as_target_network_for(self.actor, opts.target_update_rate)
    self.target_critic.set_as_target_network_for(self.critic, opts.target_update_rate)

  def run_training(self, max_num_actions, max_run_time, batch_size, batches_per_step,
                   saver_util):
    # log start time, in case we are limiting by time...
    start_time = time.time()

    # run for some max number of actions
    num_actions_taken = 0
    n = 0

    one_hot_list = util.one_hot_encode(self.obj_list)

    while True:
      rewards = []
      losses = []
      remain_obj = [ i for i in range(10)]

      # run an episode
      env.shuffle_obj() # shuffle object

      for _ in range(10):
        target_obj_idx = self.obj_list.index(random.sample(remain_obj, 1)[0])

        obj_name = env.obj_list[target_obj_idx]
        print('target object:', obj_name)

        # start a new episode
        state_1 = self.env.reset(target_obj_idx)

        # prepare data for updating replay memory at end of episode
        action_reward_state_sequence = []

        target_obj_hot = one_hot_list[target_obj_idx, :]
        done = False
        step = 0
        while not done:
          # choose action
          action = self.actor.action_given(state_1, add_noise=True)
          # take action step in env
          state_2, reward, done = self.env.step(action)
          rewards.append(reward)
          # cache for adding to replay memory
          action_reward_state_sequence.append((action, reward, np.copy(state_2)))
          # roll state for next step.
          state_1 = state_2

          step = step + 1

          # if step == opts.action_repeat_per_scene:
          if step == 1:
            done = True

        # at end of episode update replay memory
        self.replay_memory.add_episode(initial_state, action_reward_state_sequence)

        if len(remain_obj) > 0:
          # Target object remove version
          remain_obj.remove(self.obj_list[target_obj_idx])
          env.remove_obj(target_obj_idx)

          ## Random object remove version
          # rand_idx= self.obj_list.index(random.sample(remain_obj, 1)[0])
          # remain_obj.remove(rand_idx)
          # env.remove_obj(rand_idx)

        # do a training step (after waiting for buffer to fill a bit...)
        if self.replay_memory.size() > opts.replay_memory_burn_in:
          # run a set of batches
          for _ in range(batches_per_step):
            batch = self.replay_memory.batch(batch_size)
            self.actor.train(batch.state_1)
            self.critic.train(batch)

          # update target nets
          self.target_actor.update_weights()
          self.target_critic.update_weights()
          # do debug (if requested) on last batch

          # dump some stats and progress info
          stats = collections.OrderedDict()
          stats["time"] = time.time()
          stats["n"] = n
    #      stats["mean_losses"] = float(np.mean(losses))
          stats["total_reward"] = np.sum(rewards)
          stats["episode_len"] = len(rewards)
          stats["replay_memory_stats"] = self.replay_memory.current_stats()
          print("STATS %s\t%s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  json.dumps(stats)))
          sys.stdout.flush()
          n += 1

          # save if required
          if saver_util is not None:
            saver_util.save_if_required()

          # exit when finished
          num_actions_taken += len(rewards)
          if max_num_actions > 0 and num_actions_taken > max_num_actions:
            break
          if max_run_time > 0 and time.time() > start_time + max_run_time:
            break

      # emit occasional eval
      self.run_eval(1)

      # dump weights once if requested
      global DUMP_WEIGHTS
      if DUMP_WEIGHTS:
        self.debug_dump_network_weights()
        DUMP_WEIGHTS = False


  def run_eval(self, num_episodes, add_noise=False):
    """ run num_episodes of eval and output episode length and rewards """
    obj_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_obj = 10
    one_hot_list = self.one_hot_encode(obj_list, n_obj)
    obj_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(num_episodes):
      env.shuffle_obj()
      state = self.env.reset()
      num_remain_obj = random.randrange(1, 11)

      for j in range(10, num_remain_obj, -1):
        remove_object_index = random.randrange(0, j)
        env.remove_obj(obj_index[remove_object_index])
        del obj_index[remove_object_index]

      target_object_index = random.randrange(0, num_remain_obj)
      objects_name = ' ' #env.get_obj_name(obj_index[target_object_index])
      print('target object:', objects_name)

      total_reward = 0
      steps = 0
      done = False
      while not done:

        internal_state = env.get_internal_state()
        target_obj = one_hot_list[obj_index[target_object_index], :]
        action = self.actor.action_given(state, internal_state, target_obj)
        # take action step in env

        state, reward, done = self.env.step(action, obj_index[target_object_index])
        print("EVALSTEP r%s %s %s %s %s" % (i, steps, np.squeeze(action), np.linalg.norm(action), reward))
        total_reward += reward
        steps += 1

        if steps == 10:
          done = True

      print("EVAL", i, steps, total_reward)

    sys.stdout.flush()

  def debug_dump_network_weights(self):
    fn = "/tmp/weights.%s" % time.time()
    with open(fn, "w") as f:
      f.write("DUMP time %s\n" % time.time())
      for var in tf.global_variables():
        f.write("VAR %s %s\n" % (var.name, var.get_shape()))
        f.write("%s\n" % var.eval())
    print("weights written to", fn)


def main():
  config = tf.ConfigProto()
#  config.gpu_options.allow_growth = True
#  config.log_device_placement = True
  with tf.Session(config=config) as sess:
    agent = DeepDeterministicPolicyGradientAgent(env=env)

    # setup saver util and either load latest ckpt or init variables
    saver_util = None
    if opts.ckpt_dir is not None:
      saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
    else:
      sess.run(tf.global_variables_initializer())

    for v in tf.global_variables():
      print(v.name, util.shape_and_product_of(v), file=sys.stderr)
   #   print >>sys.stderr, v.name, util.shape_and_product_of(v)

    # now that we've either init'd from scratch, or loaded up a checkpoint,
    # we can do any required post init work.
    agent.post_var_init_setup()

    #opts.num_eval = 100
    # run either eval or training
    if opts.num_eval > 0:
      agent.run_eval(opts.num_eval, opts.eval_action_noise)
    else:
      agent.run_training(opts.max_num_actions, opts.max_run_time,
                         opts.batch_size, opts.batches_per_step,
                         saver_util)
      if saver_util is not None:
        saver_util.force_save()

    env.reset()  # just to flush logging, clumsy :/

if __name__ == "__main__":
  #try:
  main()
  #except:
  #  env.shutdown()
