from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import sys
import time
import json

import numpy as np
import tensorflow as tf

from model import PTBNASModel
from six.moves import xrange


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "", "")
flags.DEFINE_string("output_dir", "", "")
flags.DEFINE_string("fixed_arc", None, "")
flags.DEFINE_integer("batch_size", 25, "")
flags.DEFINE_integer("base_number", 4, "")
flags.DEFINE_integer("num_layers", 2, "")
flags.DEFINE_integer("bptt_steps", 20, "")
flags.DEFINE_integer("lstm_hidden_size", 200, "")
flags.DEFINE_float("lstm_e_keep", 1.0, "")
flags.DEFINE_float("lstm_x_keep", 1.0, "")
flags.DEFINE_float("lstm_h_keep", 1.0, "")
flags.DEFINE_float("lstm_o_keep", 1.0, "")
flags.DEFINE_boolean("lstm_l_skip", False, "")
flags.DEFINE_float("lr", 1.0, "")
flags.DEFINE_float("lr_dec_rate", 0.5, "")
flags.DEFINE_float("grad_bound", 5.0, "")
flags.DEFINE_float("temperature", None, "")
flags.DEFINE_float("l2_reg", None, "")
flags.DEFINE_float("lr_dec_min", None, "")
flags.DEFINE_float("optim_moving_average", None,
             "Use the moving average of Variables")
flags.DEFINE_float("rnn_l2_reg", None, "")
flags.DEFINE_float("rnn_slowness_reg", None, "")
flags.DEFINE_float("lr_warmup_val", None, "")
flags.DEFINE_float("reset_train_states", None, "")
flags.DEFINE_integer("lr_dec_start", 4, "")
flags.DEFINE_integer("lr_dec_every", 1, "")
flags.DEFINE_integer("avg_pool_size", 1, "")
flags.DEFINE_integer("block_size", 1, "")
flags.DEFINE_integer("rhn_depth", 4, "")
flags.DEFINE_integer("lr_warmup_steps", None, "")
flags.DEFINE_string("optim_algo", "sgd", "")
flags.DEFINE_integer("num_epochs", 300, "")
flags.DEFINE_integer("log_every", 100, "How many steps to log")
flags.DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
flags.DEFINE_integer("seed", 331, "random seed")


def get_ops(x_train, x_valid, x_test):
  """Create relevant models."""

  ops = {}

  assert FLAGS.lstm_hidden_size % FLAGS.block_size == 0, (
    "--block_size has to divide lstm_hidden_size")

  model = PTBNASModel(
    x_train,
    x_valid,
    x_test,
    rnn_l2_reg=FLAGS.rnn_l2_reg,
    rnn_slowness_reg=FLAGS.rnn_slowness_reg,
    rhn_depth=FLAGS.rhn_depth,
    fixed_arc=FLAGS.fixed_arc,
    batch_size=FLAGS.batch_size,
    bptt_steps=FLAGS.bptt_steps,
    lstm_num_layers=FLAGS.num_layers,
    lstm_hidden_size=FLAGS.lstm_hidden_size,
    lstm_e_keep=FLAGS.lstm_e_keep,
    lstm_x_keep=FLAGS.lstm_x_keep,
    lstm_h_keep=FLAGS.lstm_h_keep,
    lstm_o_keep=FLAGS.lstm_o_keep,
    lstm_l_skip=FLAGS.lstm_l_skip,
    vocab_size=10000,
    lr_init=FLAGS.lr,
    lr_dec_start=FLAGS.lr_dec_start,
    lr_dec_every=FLAGS.lr_dec_every,
    lr_dec_rate=FLAGS.lr_dec_rate,
    lr_dec_min=FLAGS.lr_dec_min,
    lr_warmup_val=FLAGS.lr_warmup_val,
    lr_warmup_steps=FLAGS.lr_warmup_steps,
    l2_reg=FLAGS.l2_reg,
    optim_moving_average=FLAGS.optim_moving_average,
    clip_mode="global",
    grad_bound=FLAGS.grad_bound,
    optim_algo="sgd",
    temperature=FLAGS.temperature,
    seed=FLAGS.seed,
    name="ptb_nas_model")

  model()

  ops = {
    "global_step": model.global_step,
    "loss": model.loss,
    "train_op": model.train_op,
    "train_ppl": model.train_ppl,
    "train_reset": model.train_reset,
    "valid_reset": model.valid_reset,
    "test_reset": model.test_reset,
    "lr": model.lr,
    "grad_norm": model.grad_norm,
    "optimizer": model.optimizer,
    'num_train_batches' : model.num_train_batches,
    'eval_every' : model.num_train_batches * FLAGS.eval_every_epochs,
    'eval_func' : model.eval_once,
  }

  return ops


def train(mode="train"):
  assert mode in ["train", "eval"], "Unknown mode '{0}'".format(mode)

  with open(FLAGS.data_path, 'rb') as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp, encoding='latin1')
    tf.logging.info("-" * 80)
    tf.logging.info("train_size: {0}".format(np.size(x_train)))
    tf.logging.info("valid_size: {0}".format(np.size(x_valid)))
    tf.logging.info(" test_size: {0}".format(np.size(x_test)))

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(x_train, x_valid, x_test)

    if FLAGS.optim_moving_average is None or mode == "eval":
      saver = tf.train.Saver(max_to_keep=10)
    else:
      saver = ops["optimizer"].swapping_saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=ops["num_train_batches"], saver=saver)

    hooks = [checkpoint_saver_hook]

    tf.logging.info("-" * 80)
    tf.logging.info("Starting session")
    with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()

        if mode == "eval":
          sess.run(ops["valid_reset"])
          ops["eval_func"](sess, "valid", verbose=True)
          sess.run(ops["test_reset"])
          ops["eval_func"](sess, "test", verbose=True)
          sys.exit(0)

        num_batches = 0
        total_tr_ppl = 0
        best_valid_ppl = 67.00
        while True:
          run_ops = [
            ops["loss"],
            ops["lr"],
            ops["grad_norm"],
            ops["train_ppl"],
            ops["train_op"],
          ]
          loss, lr, gn, tr_ppl, _ = sess.run(run_ops)
          num_batches += 1
          total_tr_ppl += loss / FLAGS.bptt_steps
          global_step = sess.run(ops["global_step"])
          actual_step = global_step
          epoch = actual_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % FLAGS.log_every == 0:
            log_string = ""
            log_string += "epoch={:<6d}".format(epoch)
            log_string += " ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.4f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<10.2f}".format(gn)
            log_string += " tr_ppl={:<8.2f}".format(
              np.exp(total_tr_ppl / num_batches))
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            tf.logging.info(log_string)

          if (FLAGS.reset_train_states is not None and
              np.random.uniform(0, 1) < FLAGS.reset_train_states):
            tf.logging.info("reset train states")
            sess.run([
              ops["train_reset"],
              ops["valid_reset"],
              ops["test_reset"],
            ])

          if actual_step % ops["eval_every"] == 0:
            sess.run([
              ops["train_reset"],
              ops["valid_reset"],
              ops["test_reset"],
            ])
            
            tf.logging.info("Epoch {}: Eval".format(epoch))
            valid_ppl = ops["eval_func"](sess, "valid")
            if valid_ppl < best_valid_ppl:
              best_valid_ppl = valid_ppl
              sess.run(ops["test_reset"])
              ops["eval_func"](sess, "test", verbose=True)

            sess.run([
              ops["train_reset"],
              ops["valid_reset"],
              ops["test_reset"],
            ])
            total_tr_ppl = 0
            num_batches = 0

            tf.logging.info("-" * 80)

          if epoch >= FLAGS.num_epochs:
            ops["eval_func"](sess, "test", verbose=True)
            break

def store_params():
  params = vars(FLAGS)['__flags']
  with open(os.path.join(FLAGS.output_dir, 'hparam.json'), 'w') as f:
    json.dump(params, f)


def main(_):
  tf.set_random_seed(FLAGS.seed)
  tf.logging.info("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    tf.logging.info("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  
  store_params()
  train(mode="train")

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

