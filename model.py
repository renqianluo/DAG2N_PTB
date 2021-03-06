from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages
from utils import count_model_params
from utils import get_train_ops
from six.moves import xrange

def ptb_input_producer(raw_data, batch_size, num_steps, shuffle=False,
                       randomize=False):
  """
  Args:
    raw_data: np tensor of size [num_words].
    batch_size: self-explained.
    num_steps: number of BPTT steps.
  """

  num_batches_per_epoch = ((np.size(raw_data) // batch_size) - 1) // num_steps
  raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

  data_len = tf.size(raw_data)
  batch_len = data_len // batch_size
  data = tf.reshape(raw_data[0 : batch_size * batch_len],
                    [batch_size, batch_len])

  epoch_size = (batch_len - 1) // num_steps
  with tf.device("/cpu:0"):
    epoch_size = tf.identity(epoch_size, name="epoch_size")
    
    if randomize:
      i = tf.random_uniform([1], minval=0, maxval=batch_len - num_steps,
                            dtype=tf.int32)
      i = tf.reduce_sum(i)
      x = tf.strided_slice(
        data, [0, i], [batch_size, i + num_steps])
      y = tf.strided_slice(
        data, [0, i + 1], [batch_size, i + num_steps + 1])
    else:
      i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
      x = tf.strided_slice(
        data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
      y = tf.strided_slice(
        data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])

    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])

  return x, y, num_batches_per_epoch


def layer_norm(x, is_training, name="layer_norm"):
  x = tf.contrib.layers.layer_norm(
    x, scope=name,
    reuse=None if is_training else True)
  return x


def batch_norm(x, is_training, name="batch_norm", decay=0.999, epsilon=1.0):
  shape = x.get_shape()[1]
  with tf.variable_scope(name, reuse=None if is_training else True):
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))

    if is_training:
      mean, variance = tf.nn.moments(x, [0])
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)

      with tf.control_dependencies([update_mean, update_variance]):
        x = scale * (x - mean) / tf.sqrt(epsilon + variance) + offset
    else:
      x = scale * (x - moving_mean) / tf.sqrt(epsilon + moving_variance) + offset
  return x

def tf_batch_norm(x, is_training):
  with tf.variable_scope('batch_norm', reuse=None if is_training else True):
    x = tf.layers.batch_normalization(
      inputs=x, axis=-1,
      momentum=0.999, epsilon=1.0, center=True,
      scale=True, training=is_training, fused=True)
  return x

def lstm(x, prev_c, prev_h, w):
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
  if initializer is None:
    initializer = tf.constant_initializer(0.0, dtype=tf.float32)
  return tf.get_variable(name, shape, initializer=initializer)


class PTBNASModel(object):
  def __init__(self,
               x_train,
               x_valid,
               x_test,
               num_funcs=4,
               rnn_l2_reg=None,
               rnn_slowness_reg=None,
               rhn_depth=2,
               fixed_arc=None,
               base_number=4,
               batch_size=32,
               bptt_steps=25,
               lstm_num_layers=2,
               lstm_hidden_size=32,
               lstm_e_keep=1.0,
               lstm_x_keep=1.0,
               lstm_h_keep=1.0,
               lstm_o_keep=1.0,
               lstm_l_skip=False,
               vocab_size=10000,
               lr_warmup_val=None,
               lr_warmup_steps=None,
               lr_init=1.0,
               lr_dec_start=4,
               lr_dec_every=1,
               lr_dec_rate=0.5,
               lr_dec_min=None,
               l2_reg=None,
               clip_mode="global",
               grad_bound=5.0,
               optim_algo=None,
               optim_moving_average=None,
               temperature=None,
               name="ptb_lstm",
               seed=None,
               *args,
               **kwargs):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    tf.logging.info("-" * 80)
    tf.logging.info("Build model {}".format(name))

    self.num_funcs = num_funcs
    self.rnn_l2_reg = rnn_l2_reg
    self.rnn_slowness_reg = rnn_slowness_reg
    self.rhn_depth = rhn_depth
    self.fixed_arc = fixed_arc
    self.base_number = base_number
    self.num_nodes = 2 * self.base_number - 1
    self.batch_size = batch_size
    self.bptt_steps = bptt_steps
    self.lstm_num_layers = lstm_num_layers
    self.lstm_hidden_size = lstm_hidden_size
    self.lstm_e_keep = lstm_e_keep
    self.lstm_x_keep = lstm_x_keep
    self.lstm_h_keep = lstm_h_keep
    self.lstm_o_keep = lstm_o_keep
    self.lstm_l_skip = lstm_l_skip
    self.vocab_size = vocab_size
    self.lr_warmup_val = lr_warmup_val
    self.lr_warmup_steps = lr_warmup_steps
    self.lr_init = lr_init
    self.lr_dec_min = lr_dec_min
    self.l2_reg = l2_reg
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound

    self.optim_algo = optim_algo
    self.optim_moving_average = optim_moving_average
    self.temperature = temperature

    self.name = name
    self.seed = seed
    
    self.global_step = None
    self.valid_loss = None
    self.test_loss = None

    tf.logging.info("Build data ops")
    # training data
    self.x_train, self.y_train, self.num_train_batches = ptb_input_producer(
      x_train, self.batch_size, self.bptt_steps)
    self.y_train = tf.reshape(self.y_train, [self.batch_size * self.bptt_steps])

    self.lr_dec_start = lr_dec_start * self.num_train_batches
    self.lr_dec_every = lr_dec_every * self.num_train_batches
    self.lr_dec_rate = lr_dec_rate

    # valid data
    self.x_valid, self.y_valid, self.num_valid_batches = ptb_input_producer(
      np.copy(x_valid), self.batch_size, self.bptt_steps)
    self.y_valid = tf.reshape(self.y_valid, [self.batch_size * self.bptt_steps])


    # test data
    self.x_test, self.y_test, self.num_test_batches = ptb_input_producer(
      x_test, 1, 1)
    self.y_test = tf.reshape(self.y_test, [1])

    self.x_valid_raw = x_valid


  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None, "TF op self.global_step not defined."
    global_step = sess.run(self.global_step)
    tf.logging.info("Eval at {}".format(global_step))
   
    if eval_set == "valid":
      assert self.valid_loss is not None, "TF op self.valid_loss is not defined."
      num_batches = self.num_valid_batches
      loss_op = self.valid_loss
      reset_op = self.valid_reset
      batch_size = self.batch_size
      bptt_steps = self.bptt_steps
    elif eval_set == "test":
      assert self.test_loss is not None, "TF op self.test_loss is not defined."
      num_batches = self.num_test_batches
      loss_op = self.test_loss
      reset_op = self.test_reset
      batch_size = 1
      bptt_steps = 1
    else:
      raise ValueError("Unknown eval_set '{}'".format(eval_set))

    sess.run(reset_op)
    total_loss = 0
    for batch_id in range(num_batches):
      curr_loss = sess.run(loss_op, feed_dict=feed_dict)
      total_loss += curr_loss #np.minimum(curr_loss, 10.0 * bptt_steps * batch_size)
      ppl_sofar = np.exp(total_loss / (bptt_steps * batch_size * (batch_id + 1)))
      if verbose and (batch_id + 1) % 1000 == 0:
        tf.logging.info("{:<5d} {:<6.2f}".format(batch_id + 1, ppl_sofar))
    if verbose:
      tf.logging.info("")
    log_ppl = total_loss / (num_batches * batch_size * bptt_steps)
    ppl = np.exp(np.minimum(log_ppl, 10.0))
    sess.run(reset_op)
    tf.logging.info("{}_total_loss: {:<6.2f}".format(eval_set, total_loss))
    tf.logging.info("{}_log_ppl: {:<6.2f}".format(eval_set, log_ppl))
    tf.logging.info("{}_ppl: {:<6.2f}".format(eval_set, ppl))
    return ppl

  def _build_train(self):
    tf.logging.info("Build train graph")
    all_h, self.train_reset = self._model(self.x_train, True, False)
    log_probs = self._get_log_probs(
      all_h, self.y_train, batch_size=self.batch_size, is_training=True)
    total_log_probs = tf.reduce_sum(log_probs)
    self.loss = total_log_probs / tf.to_float(self.batch_size)
    self.train_ppl = tf.exp(tf.reduce_mean(log_probs))

    tf_variables = [
      var for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    tf.logging.info("-" * 80)
    tf.logging.info("Model has {} parameters".format(self.num_vars))

    loss = self.loss
    if self.rnn_l2_reg is not None:
      loss += (self.rnn_l2_reg * tf.reduce_sum(all_h ** 2) /
               tf.to_float(self.batch_size))
    if self.rnn_slowness_reg is not None:
      loss += (self.rnn_slowness_reg * self.all_h_diff /
               tf.to_float(self.batch_size))
    
    self.global_step = tf.train.get_or_create_global_step()
    (self.train_op,
     self.lr,
     self.grad_norm,
     self.optimizer,
     self.grad_norms) = get_train_ops(
       loss,
       tf_variables,
       self.global_step,
       clip_mode=self.clip_mode,
       grad_bound=self.grad_bound,
       l2_reg=self.l2_reg,
       lr_warmup_val=self.lr_warmup_val,
       lr_warmup_steps=self.lr_warmup_steps,
       lr_init=self.lr_init,
       lr_dec_start=self.lr_dec_start,
       lr_dec_every=self.lr_dec_every,
       lr_dec_rate=self.lr_dec_rate,
       lr_dec_min=self.lr_dec_min,
       optim_algo=self.optim_algo,
       moving_average=self.optim_moving_average,
       get_grad_norms=True,
     )

  def _get_log_probs(self, all_h, labels, batch_size=None, is_training=False):
    logits = tf.matmul(all_h, self.w_emb, transpose_b=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
    return log_probs

  def _build_valid(self):
    tf.logging.info("-" * 80)
    tf.logging.info("Build valid graph")
    all_h, self.valid_reset = self._model(self.x_valid, False, False)
    all_h = tf.stop_gradient(all_h)
    log_probs = self._get_log_probs(all_h, self.y_valid)
    self.valid_loss = tf.reduce_sum(log_probs)


  def _build_test(self):
    tf.logging.info("-" * 80)
    tf.logging.info("Build test graph")
    all_h, self.test_reset = self._model(self.x_test, False, True)
    all_h = tf.stop_gradient(all_h)
    log_probs = self._get_log_probs(all_h, self.y_test)
    self.test_loss = tf.reduce_sum(log_probs)

  def _rhn_fixed(self, x, prev_s, w_prev, w_skip, is_training,
                 x_mask=None, s_mask=None):
    batch_size = prev_s.get_shape()[0].value
    if is_training:
      assert x_mask is not None, "x_mask is None"
      assert s_mask is not None, "s_mask is None"
      ht = tf.matmul(tf.concat([x * x_mask, prev_s * s_mask], axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([x, prev_s], axis=1), w_prev)
    with tf.variable_scope("rhn_layer_0"):
      ht = layer_norm(ht, is_training)
    h, t = tf.split(ht, 2, axis=1)

    if self.sample_arc[0] == 0:
      h = tf.tanh(h)
    elif self.sample_arc[0] == 1:
      h = tf.nn.relu(h)
    elif self.sample_arc[0] == 2:
      h = tf.identity(h)
    elif self.sample_arc[0] == 3:
      h = tf.sigmoid(h)
    else:
      raise ValueError("Unknown func_idx {}".format(self.sample_arc[0]))
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    start_idx = 1
    used = np.zeros([self.rhn_depth], dtype=np.int32)
    for rhn_layer_id in range(1, self.rhn_depth):
      with tf.variable_scope("rhn_layer_{}".format(rhn_layer_id)):
        prev_idx = self.sample_arc[start_idx]
        func_idx = self.sample_arc[start_idx + 1]
        used[prev_idx] = 1
        prev_s = layers[prev_idx]
        if is_training:
          ht = tf.matmul(prev_s * s_mask, w_skip[rhn_layer_id])
        else:
          ht = tf.matmul(prev_s, w_skip[rhn_layer_id])
        ht = layer_norm(ht, is_training)
        h, t = tf.split(ht, 2, axis=1)

        if func_idx == 0:
          h = tf.tanh(h)
        elif func_idx == 1:
          h = tf.nn.relu(h)
        elif func_idx == 2:
          h = tf.identity(h)
        elif func_idx == 3:
          h = tf.sigmoid(h)
        else:
          raise ValueError("Unknown func_idx {}".format(func_idx))

        t = tf.sigmoid(t)
        s = prev_s + t * (h - prev_s)
        layers.append(s)
        start_idx += 2

    layers = [prev_layer for u, prev_layer in zip(used, layers) if u == 0]
    layers = tf.add_n(layers) / np.sum(1.0 - used)
    layers.set_shape([batch_size, self.lstm_hidden_size])

    return layers


  def _model(self, x, is_training, is_test, should_carry=True):
    if is_test:
      start_h = self.test_start_h
      num_steps = 1
      batch_size = 1
    else:
      start_h = self.start_h
      num_steps = self.bptt_steps
      batch_size = self.batch_size

    all_h = tf.TensorArray(tf.float32, size=num_steps, infer_shape=True)
    embedding = tf.nn.embedding_lookup(self.w_emb, x)
    if is_training:
      def _gen_mask(shape, keep_prob):
        _mask = tf.random_uniform(shape, dtype=tf.float32)
        _mask = tf.floor(_mask + keep_prob) / keep_prob
        return _mask

      # variational dropout in the embedding layer
      e_mask = _gen_mask([batch_size, num_steps], self.lstm_e_keep)
      first_e_mask = e_mask
      zeros = tf.zeros_like(e_mask)
      ones = tf.ones_like(e_mask)
      r = [tf.constant([[False]] * batch_size, dtype=tf.bool)]  # more zeros to e_mask
      for step in range(1, num_steps):
        should_zero = tf.logical_and(
          tf.equal(x[:, :step], x[:, step:step+1]),
          tf.equal(e_mask[:, :step], 0))
        should_zero = tf.reduce_any(should_zero, axis=1, keep_dims=True)
        r.append(should_zero)
      r = tf.concat(r, axis=1)
      e_mask = tf.where(r, tf.zeros_like(e_mask), e_mask)
      e_mask = tf.reshape(e_mask, [batch_size, num_steps, 1])
      embedding *= e_mask
      # variational dropout in the hidden layers
      x_mask, h_mask = [], []
      for layer_id in range(self.lstm_num_layers):
        x_mask.append(_gen_mask([batch_size, self.lstm_hidden_size], self.lstm_x_keep))
        h_mask.append(_gen_mask([batch_size, self.lstm_hidden_size], self.lstm_h_keep))

      # variational dropout in the output layer
      o_mask = _gen_mask([batch_size, self.lstm_hidden_size], self.lstm_o_keep)

    def condition(step, *args):
      return tf.less(step, num_steps)

    def body(step, prev_h, all_h):
      with tf.variable_scope(self.name):
        next_h = []
        for layer_id, (p_h, w_prev, w_skip) in enumerate(zip(prev_h, self.w_prev, self.w_skip)):
          with tf.variable_scope("layer_{}".format(layer_id)):
            if layer_id == 0:
              inputs = embedding[:, step, :]
            else:
              inputs = next_h[-1]
              
            curr_h = self._rhn_fixed(
              inputs, p_h, w_prev, w_skip, is_training,
              x_mask=x_mask[layer_id] if is_training else None,
              s_mask=h_mask[layer_id] if is_training else None)

            if self.lstm_l_skip:  #skip connections
              curr_h += inputs

            next_h.append(curr_h)

        out_h = next_h[-1]
        if is_training:
          out_h *= o_mask
        all_h = all_h.write(step, out_h)
      return step + 1, next_h, all_h
    
    loop_vars = [tf.constant(0, dtype=tf.int32), start_h, all_h]
    loop_outputs = tf.while_loop(condition, body, loop_vars, back_prop=True)
    next_h = loop_outputs[-2]
    all_h = loop_outputs[-1].stack()
    all_h_diff = (all_h[1:, :, :] - all_h[:-1, :, :]) ** 2
    self.all_h_diff = tf.reduce_sum(all_h_diff)
    all_h = tf.transpose(all_h, [1, 0, 2])#from [ts,bs,h] to [bs,ts,h]
    all_h = tf.reshape(all_h, [batch_size * num_steps, self.lstm_hidden_size])
    
    carry_states = []
    reset_states = []
    for layer_id, (s_h, n_h) in enumerate(zip(start_h, next_h)):
      reset_states.append(tf.assign(s_h, tf.zeros_like(s_h), use_locking=True))
      carry_states.append(tf.assign(s_h, tf.stop_gradient(n_h), use_locking=True))

    if should_carry:
      with tf.control_dependencies(carry_states):
        all_h = tf.identity(all_h)

    return all_h, reset_states

  def _build_params(self):
    if self.lstm_hidden_size <= 300:
      init_range = 0.1
    elif self.lstm_hidden_size <= 400:
      init_range = 0.05
    else:
      init_range = 0.04
    initializer = tf.random_uniform_initializer(
      minval=-init_range, maxval=init_range)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope("rnn"):
        self.w_prev, self.w_skip = [], []
        for layer_id in range(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w_prev = tf.get_variable("w_prev", [2 * self.lstm_hidden_size,
                                                  2 * self.lstm_hidden_size])
            w_skip = [None]
            for rhn_layer_id in range(1, self.rhn_depth):
              with tf.variable_scope("layer_{}".format(rhn_layer_id)):
                w = tf.get_variable("w", [self.lstm_hidden_size,
                                          2 * self.lstm_hidden_size])
                w_skip.append(w)
            self.w_prev.append(w_prev)
            self.w_skip.append(w_skip)

      with tf.variable_scope("embedding"):
        self.w_emb = tf.get_variable(
          "w", [self.vocab_size, self.lstm_hidden_size])

      with tf.variable_scope("starting_states"):
        zeros = np.zeros(
          [self.batch_size, self.lstm_hidden_size], dtype=np.float32)
        zeros_one_instance = np.zeros(
          [1, self.lstm_hidden_size], dtype=np.float32)

        self.start_h, self.test_start_h = [], []
        for _ in range(self.lstm_num_layers):
          self.start_h.append(tf.Variable(zeros, trainable=False))
          self.test_start_h.append(tf.Variable(zeros_one_instance,
                                               trainable=False))

  def __call__(self):
    self.sample_arc = np.array(
      [x for x in self.fixed_arc.split(' ') if x], dtype=np.int32)

    self._build_params()
    self._build_train()
    self._build_valid()
    self._build_test()
