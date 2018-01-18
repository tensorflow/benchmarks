# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import print_function

import argparse
from collections import namedtuple
import math
import multiprocessing
import os
import threading
import time

from absl import flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import benchmark_storage
import cnn_util
import convnet_builder
import datasets
import variable_mgr
import variable_mgr_util
from cnn_util import log_fn
from cnn_util import ParamSpec
from models import model_config
from platforms import util as platforms_util


_DEFAULT_NUM_BATCHES = 100


# _DEFAULT_PARAMS maps from each parameter's name to its ParamSpec. For each
# parameter listed here, a command line flag will be defined for
# tf_cnn_benchmarks.py.
_DEFAULT_PARAMS = {
    'model':
        ParamSpec('string', 'trivial', 'name of the model to run'),

    # The code will first check if it's running under benchmarking mode
    # or evaluation mode, depending on 'eval':
    # Under the evaluation mode, this script will read a saved model,
    #   and compute the accuracy of the model against a validation dataset.
    #   Additional ops for accuracy and top_k predictors are only used under
    #   this mode.
    # Under the benchmarking mode, user can specify whether nor not to use
    #   the forward-only option, which will only compute the loss function.
    #   forward-only cannot be enabled with eval at the same time.
    'eval':
        ParamSpec('boolean', False, 'whether use eval or benchmarking'),
    'eval_interval_secs':
        ParamSpec('integer', 0,
                  'How often to run eval on saved checkpoints. Usually the '
                  'same as save_model_secs from the corresponding training '
                  'run. Pass 0 to eval only once.'),
    'forward_only':
        ParamSpec('boolean', False,
                  'whether use forward-only or training for benchmarking'),
    'print_training_accuracy':
        ParamSpec('boolean', False,
                  'whether to calculate and print training accuracy during '
                  'training'),
    'batch_size':
        ParamSpec('integer', 0, 'batch size per compute device'),
    'batch_group_size':
        ParamSpec('integer', 1,
                  'number of groups of batches processed in the image '
                  'producer.'),
    'num_batches':
        ParamSpec('integer', None, 'number of batches to run, excluding '
                  'warmup. Defaults to %d' % _DEFAULT_NUM_BATCHES),
    'num_epochs':
        ParamSpec('float', None, 'number of epochs to run, excluding warmup. '
                  'This and --num_batches cannot both be specified.'),
    'num_warmup_batches':
        ParamSpec('integer', None, 'number of batches to run before timing'),
    'autotune_threshold':
        ParamSpec('integer', None, 'The autotune threshold for the models'),
    'num_gpus':
        ParamSpec('integer', 1, 'the number of GPUs to run on'),
    'gpu_indices':
        ParamSpec('string', '', 'indices of worker GPUs in ring order'),
    'display_every':
        ParamSpec('integer', 10,
                  'Number of local steps after which progress is printed out'),
    'data_dir':
        ParamSpec('string', None,
                  'Path to dataset in TFRecord format (aka Example '
                  'protobufs). If not specified, synthetic data will be '
                  'used.'),
    'data_name':
        ParamSpec('string', None,
                  'Name of dataset: imagenet or cifar10. If not specified, it '
                  'is automatically guessed based on data_dir.'),
    'resize_method':
        ParamSpec('string', 'bilinear',
                  'Method for resizing input images: crop, nearest, bilinear, '
                  'bicubic, area, or round_robin. The `crop` mode requires '
                  'source images to be at least as large as the network input '
                  'size. The `round_robin` mode applies different resize '
                  'methods based on position in a batch in a round-robin '
                  'fashion. Other modes support any sizes and apply random '
                  'bbox distortions before resizing (even with '
                  'distortions=False).'),
    'distortions':
        ParamSpec('boolean', True,
                  'Enable/disable distortions during image preprocessing. '
                  'These include bbox and color distortions.'),
    'use_datasets':
        ParamSpec('boolean', True, 'Enable use of datasets for input pipeline'),
    'gpu_thread_mode':
        ParamSpec('string', 'gpu_private',
                  'Methods to assign GPU host work to threads. '
                  'global: all GPUs and CPUs share the same global threads; '
                  'gpu_private: a private threadpool for each GPU; '
                  'gpu_shared: all GPUs share the same threadpool.'),
    'per_gpu_thread_count':
        ParamSpec('integer', 0, 'The number of threads to use for GPU.'
                  'Only valid when gpu_thread_mode is not global.'),
    'cache_data':
        ParamSpec('boolean', False,
                  'Enable use of a special datasets pipeline that reads a '
                  'single TFRecord into memory and repeats it infinitely many '
                  'times. The purpose of this flag is to make it possible '
                  'to write regression tests that are not bottlenecked by CNS '
                  'throughput.'),
    'local_parameter_device':
        ParamSpec('string', 'gpu',
                  'Device to use as parameter server: cpu or gpu. For '
                  'distributed training, it can affect where caching of '
                  'variables happens.'),
    'device':
        ParamSpec('string', 'gpu', 'Device to use for computation: cpu or gpu'),
    'data_format':
        ParamSpec('string', 'NCHW',
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).'),
    'num_intra_threads':
        ParamSpec('integer', 1,
                  'Number of threads to use for intra-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'num_inter_threads':
        ParamSpec('integer', 0,
                  'Number of threads to use for inter-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'trace_file':
        ParamSpec('string', '',
                  'Enable TensorFlow tracing and write trace to this file.'),
    'graph_file':
        ParamSpec('string', None,
                  'Write the model\'s graph definition to this file. Defaults '
                  'to binary format unless filename ends in "txt".'),
    'optimizer':
        ParamSpec('string', 'sgd',
                  'Optimizer to use: momentum or sgd or rmsprop'),
    'learning_rate':
        ParamSpec('float', None, 'Initial learning rate for training.'),
    'piecewise_learning_rate_schedule':
        ParamSpec('string', None,
                  'Specifies a piecewise learning rate schedule based on the '
                  'number of epochs. This is the form LR0;E1;LR1;...;En;LRn, '
                  'where each LRi is a learning rate and each Ei is an epoch '
                  'indexed from 0. The learning rate is LRi if the '
                  'E(i-1) <= current_epoch < Ei. For example, if this '
                  'paramater is 0.3;10;0.2;25;0.1, the learning rate is 0.3 '
                  'for the first 10 epochs, then is 0.2 for the next 15 '
                  'epochs, then is 0.1 until training ends.'),
    'num_epochs_per_decay':
        ParamSpec('float', 0,
                  'Steps after which learning rate decays. If 0, the learning '
                  'rate does not decay.'),
    'learning_rate_decay_factor':
        ParamSpec('float', 0,
                  'Learning rate decay factor. Decay by this factor every '
                  '`num_epochs_per_decay` epochs. If 0, learning rate does '
                  'not decay.'),
    'minimum_learning_rate':
        ParamSpec('float', 0,
                  'The minimum learning rate. The learning rate will '
                  'never decay past this value. Requires `learning_rate`, '
                  '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                  'be set.'),
    'momentum':
        ParamSpec('float', 0.9, 'Momentum for training.'),
    'rmsprop_decay':
        ParamSpec('float', 0.9, 'Decay term for RMSProp.'),
    'rmsprop_momentum':
        ParamSpec('float', 0.9, 'Momentum in RMSProp.'),
    'rmsprop_epsilon':
        ParamSpec('float', 1.0, 'Epsilon term for RMSProp.'),
    'gradient_clip':
        ParamSpec('float', None,
                  'Gradient clipping magnitude. Disabled by default.'),
    'weight_decay':
        ParamSpec('float', 0.00004, 'Weight decay factor for training.'),
    'gpu_memory_frac_for_testing':
        ParamSpec('float', 0,
                  'If non-zero, the fraction of GPU memory that will be used. '
                  'Useful for testing the benchmark script, as this allows '
                  'distributed mode to be run on a single machine. For '
                  'example, if there are two tasks, each can be allocated '
                  '~40 percent of the memory on a single machine'),
    'use_tf_layers':
        ParamSpec('boolean', True,
                  'If True, use tf.layers for neural network layers. This '
                  'should not affect performance or accuracy in any way.'),
    'tf_random_seed':
        ParamSpec('integer', 1234,
                  'The TensorFlow random seed. Useful for debugging NaNs, as '
                  'this can be set to various values to see if the NaNs '
                  'depend on the seed.'),
    'debugger':
        ParamSpec('string', None,
                  'If set, use the TensorFlow debugger. If set to "cli", use '
                  'the local CLI debugger. Otherwise, this must be in the form '
                  'hostname:port (e.g., localhost:7007), in which case the '
                  'experimental TensorBoard debugger will be used'),

    # Performance tuning parameters.
    'winograd_nonfused':
        ParamSpec('boolean', True,
                  'Enable/disable using the Winograd non-fused algorithms.'),
    'sync_on_finish':
        ParamSpec('boolean', False,
                  'Enable/disable whether the devices are synced after each '
                  'step.'),
    'staged_vars':
        ParamSpec('boolean', False,
                  'whether the variables are staged from the main '
                  'computation'),
    'force_gpu_compatible':
        ParamSpec('boolean', True,
                  'whether to enable force_gpu_compatible in GPU_Options'),
    'xla':
        ParamSpec('boolean', False, 'whether to enable XLA'),
    'fuse_decode_and_crop':
        ParamSpec('boolean', True,
                  'Fuse decode_and_crop for image preprocessing.'),
    'distort_color_in_yiq':
        ParamSpec('boolean', True,
                  'Distort color of input images in YIQ space.'),
    'enable_layout_optimizer':
        ParamSpec('boolean', False, 'whether to enable layout optimizer'),
    # Performance tuning specific to MKL.
    'mkl':
        ParamSpec('boolean', False, 'If true, set MKL environment variables.'),
    'kmp_blocktime':
        ParamSpec('integer', 30,
                  'The time, in milliseconds, that a thread should wait, '
                  'after completing the execution of a parallel region, '
                  'before sleeping'),
    'kmp_affinity':
        ParamSpec('string', 'granularity=fine,verbose,compact,1,0',
                  'Restricts execution of certain threads (virtual execution '
                  'units) to a subset of the physical processing units in a '
                  'multiprocessor computer.'),
    'kmp_settings':
        ParamSpec('integer', 1, 'If set to 1, MKL settings will be printed.'),

    # fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
    'use_fp16':
        ParamSpec('boolean', False,
                  'Use 16-bit floats for certain tensors instead of 32-bit '
                  'floats. This is currently experimental.'),
    # TODO(reedwm): The default loss scale of 128 causes most models to diverge
    # on the second step with synthetic data. Changing the tf.set_random_seed
    # call to tf.set_random_seed(1235) or most other seed values causes the
    # issue not to occur.
    'fp16_loss_scale':
        ParamSpec('float', None,
                  'If fp16 is enabled, the loss is multiplied by this amount '
                  'right before gradients are computed, then each gradient '
                  'is divided by this amount. Mathematically, this has no '
                  'effect, but it helps avoid fp16 underflow. Set to 1 to '
                  'effectively disable.'),
    'fp16_vars':
        ParamSpec('boolean', False,
                  'If fp16 is enabled, also use fp16 for variables. If False, '
                  'the variables are stored in fp32 and casted to fp16 when '
                  'retrieved.  Recommended to leave as False.'),
    'fp16_enable_auto_loss_scale':
        ParamSpec('boolean', False,
                  'If True and use_fp16 is True, automatically adjust the loss'
                  ' scale during training.'),
    'fp16_inc_loss_scale_every_n':
        ParamSpec('integer', 1000,
                  'If fp16 is enabled and fp16_enable_auto_loss_scale is True,'
                  ' increase the loss scale every n steps.'),

    # The method for managing variables:
    #   parameter_server: variables are stored on a parameter server that holds
    #       the master copy of the variable. In local execution, a local device
    #       acts as the parameter server for each variable; in distributed
    #       execution, the parameter servers are separate processes in the
    #       cluster.
    #       For each step, each tower gets a copy of the variables from the
    #       parameter server, and sends its gradients to the param server.
    #   replicated: each GPU has its own copy of the variables. To apply
    #       gradients, an all_reduce algorithm or or regular cross-device
    #       aggregation is used to replicate the combined gradients to all
    #       towers (depending on all_reduce_spec parameter setting).
    #   independent: each GPU has its own copy of the variables, and gradients
    #       are not shared between towers. This can be used to check performance
    #       when no data is moved between GPUs.
    #   distributed_replicated: Distributed training only. Each GPU has a copy
    #       of the variables, and updates its copy after the parameter servers
    #       are all updated with the gradients from all servers. Only works with
    #       cross_replica_sync=true. Unlike 'replicated', currently never uses
    #       nccl all-reduce for replicating within a server.
    #   distributed_all_reduce: Distributed training where all replicas run
    #       in a single session, using all-reduce to mutally reduce the
    #       gradients.  Uses no parameter servers.  When there is only one
    #       worker, this is the same as replicated.
    'variable_update':
        ParamSpec('string', 'parameter_server',
                  'The method for managing variables: parameter_server, '
                  'replicated, distributed_replicated, independent, '
                  'distributed_all_reduce'),
    'all_reduce_spec':
        ParamSpec('string', None,
                  'A specification of the all_reduce algorithm to be used for '
                  'reducing gradients.  For more details, see '
                  'parse_all_reduce_spec in variable_mgr.py.  An '
                  'all_reduce_spec has BNF form:\n'
                  'int ::= positive whole number\n'
                  'g_int ::= int[KkMGT]?\n'
                  'alg_spec ::= alg | alg#int\n'
                  'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                  'spec ::= range_spec | range_spec:g_int:range_spec\n'
                  'NOTE: not all syntactically correct constructs are '
                  'supported.\n\n'
                  'Examples:\n '
                  '"xring" == use one global ring reduction for all '
                  'tensors\n'
                  '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                  '"nccl" == use NCCL to locally reduce all tensors.  '
                  'Limited to 1 worker.\n'
                  '"nccl/xring" == locally (to one worker) reduce values '
                  'using NCCL then ring reduce across workers.\n'
                  '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                  'size up to 32kB, then xring for larger tensors.'),

    # If variable_update==distributed_all_reduce then it may be advantageous
    # to aggregate small tensors into one prior to reduction.  These parameters
    # control that aggregation.
    'agg_small_grads_max_bytes':
        ParamSpec('integer', 0, 'If > 0, try to aggregate tensors of less '
                  'than this number of bytes prior to all-reduce.'),
    'agg_small_grads_max_group':
        ParamSpec('integer', 10, 'When aggregating small tensors for '
                  'all-reduce do not aggregate more than this many into one '
                  'new tensor.'),

    # Distributed training parameters.
    'job_name':
        ParamSpec('string', '',
                  'One of "ps", "worker", "".  Empty for local training'),
    'ps_hosts':
        ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'worker_hosts':
        ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'controller_host':
        ParamSpec('string', None, 'optional controller host'),
    'task_index':
        ParamSpec('integer', 0, 'Index of task within the job'),
    'server_protocol':
        ParamSpec('string', 'grpc', 'protocol for servers'),
    'cross_replica_sync':
        ParamSpec('boolean', True, ''),

    # Summary and Save & load checkpoints.
    'summary_verbosity':
        ParamSpec(
            'integer', 0, 'Verbosity level for summary ops. '
            '  level 0: disable any summary. '
            '  level 1: small and fast ops, e.g.: learning_rate, total_loss.'
            '  level 2: medium-cost ops, e.g. histogram of all gradients.'
            '  level 3: expensive ops: images and histogram of each gradient.'),
    'save_summaries_steps':
        ParamSpec('integer', 0,
                  'How often to save summaries for trained models. Pass 0 to '
                  'disable summaries.'),
    'save_model_secs':
        ParamSpec('integer', 0,
                  'How often to save trained models. Pass 0 to disable '
                  'checkpoints.'),
    'train_dir':
        ParamSpec('string', None,
                  'Path to session checkpoints. Pass None to disable saving '
                  'checkpoint at the end.'),
    'eval_dir':
        ParamSpec('string', '/tmp/tf_cnn_benchmarks/eval',
                  'Directory where to write eval event logs.'),
    'result_storage':
        ParamSpec('string', None,
                  'Specifies storage option for benchmark results. None means '
                  'results won\'t be stored. `cbuild_benchmark_datastore` '
                  'means results will be stored in cbuild datastore (note: '
                  'this option requires special permissions and meant to be '
                  'used from cbuilds).'),
}
_DEFAULT_PARAMS.update(platforms_util.get_platform_params())


def define_flags():
  """Define a command line flag for each ParamSpec in _DEFAULT_PARAMS."""
  define_flag = {
      'boolean': flags.DEFINE_boolean,
      'float': flags.DEFINE_float,
      'integer': flags.DEFINE_integer,
      'string': flags.DEFINE_string,
  }
  for name, param_spec in six.iteritems(_DEFAULT_PARAMS):
    if param_spec.flag_type not in define_flag:
      raise ValueError('Unknown flag_type %s' % param_spec.flag_type)
    else:
      define_flag[param_spec.flag_type](name, param_spec.default_value,
                                        param_spec.description)
      flags.declare_key_flag(name)


FLAGS = flags.FLAGS


class GlobalStepWatcher(threading.Thread):
  """A helper class for globe_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op, start_at_global_step,
               end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def num_steps(self):
    return self.finish_step - self.start_step

  def elapsed_time(self):
    return self.finish_time - self.start_time


class CheckpointNotFoundException(Exception):
  pass


def get_data_type(params):
  """Returns BenchmarkCNN's data type as determined by use_fp16.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  return tf.float16 if params.use_fp16 else tf.float32


def loss_function(logits, labels, aux_logits):
  """Loss function."""
  with tf.name_scope('xentropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  if aux_logits is not None:
    with tf.name_scope('aux_xentropy'):
      aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=aux_logits, labels=labels)
      aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
      loss = tf.add_n([loss, aux_loss])
  return loss


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = params.num_intra_threads
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
  if params.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        params.gpu_memory_frac_for_testing)
  if params.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if params.enable_layout_optimizer:
    config.graph_options.rewrite_options.layout_optimizer = (
        rewriter_config_pb2.RewriterConfig.ON)

  return config


def get_mode_from_params(params):
  """Returns the mode in which this script is running.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Raises:
    ValueError: Unsupported params settings.
  """
  if params.forward_only and params.eval:
    raise ValueError('Only one of forward_only and eval parameters is true')

  if params.eval:
    return 'evaluation'
  if params.forward_only:
    return 'forward-only'
  return 'training'


def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       image_producer,
                       params,
                       summary_op=None):
  """Advance one step of benchmarking."""
  if trace_filename and step == -1:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
  else:
    run_options = None
    run_metadata = None
  summary_str = None
  start_time = time.time()
  if summary_op is None:
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  else:
    (results, summary_str) = sess.run(
        [fetches, summary_op], options=run_options, run_metadata=run_metadata)

  if not params.forward_only:
    lossval = results['total_loss']
  else:
    lossval = 0.
  image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if step >= 0 and (step == 0 or (step + 1) % params.display_every == 0):
    log_str = '%i\t%s\t%.3f' % (
        step + 1, get_perf_timing_str(batch_size, step_train_times), lossval)
    if 'top_1_accuracy' in results:
      log_str += '\t%.3f\t%.3f' % (results['top_1_accuracy'],
                                   results['top_5_accuracy'])
    log_fn(log_str)
  if trace_filename and step == -1:
    log_fn('Dumping trace to %s' % trace_filename)
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_dir = os.path.dirname(trace_filename)
    if not gfile.Exists(trace_dir):
      gfile.MakeDirs(trace_dir)
    with gfile.Open(trace_filename, 'w') as trace_file:
      trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
  return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))
  else:
    return 'images/sec: %.1f' % speed_mean


def load_checkpoint(saver, sess, ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      # Restores from checkpoint with relative path.
      model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
      global_step = 0
    else:
      global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)
    log_fn('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
    return global_step
  else:
    raise CheckpointNotFoundException('No checkpoint file found.')


# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in _DEFAULT_PARAMS.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from _DEFAULT_PARAMS, rather than constructing
# Params directly.
Params = namedtuple('Params', _DEFAULT_PARAMS.keys())  # pylint: disable=invalid-name


def make_params(**kwargs):
  """Create a Params tuple for BenchmarkCNN from kwargs.

  Default values are filled in from _DEFAULT_PARAMS.

  Args:
    **kwargs: kwarg values will override the default values.
  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Create a (name: default_value) map from PARAMS.
  default_kwargs = {
      name: _DEFAULT_PARAMS[name].default_value
      for name in _DEFAULT_PARAMS
  }
  return Params(**default_kwargs)._replace(**kwargs)


def make_params_from_flags():
  """Create a Params tuple for BenchmarkCNN from FLAGS.

  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Collect (name: value) pairs for FLAGS with matching names in
  # _DEFAULT_PARAMS.
  flag_values = {name: getattr(FLAGS, name) for name in _DEFAULT_PARAMS.keys()}
  return Params(**flag_values)


def get_num_batches_and_epochs(params, batch_size, num_examples_per_epoch):
  """Returns the number of batches and epochs to run for.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    batch_size: The number of images per step.
    num_examples_per_epoch: The number of images in a single epoch.

  Returns:
    num_batches: The number of batches to run for.
    num_epochs: The number of epochs to run for. This might be slightly
      smaller than params.num_epochs if specified, because the number of batches
      must be an integer.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  if params.num_batches and params.num_epochs:
    raise ValueError('At most one of --num_batches and --num_epochs may be '
                     'specified.')
  if params.num_epochs:
    num_batches = int(float(params.num_epochs) * num_examples_per_epoch /
                      batch_size)
  else:
    num_batches = params.num_batches or _DEFAULT_NUM_BATCHES
  num_epochs = num_batches * batch_size / float(num_examples_per_epoch)
  return (num_batches, num_epochs)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
  """Returns a piecewise learning rate tensor.

  Args:
    piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
      parameter
    global_step: Scalar tensor representing the global step.
    num_batches_per_epoch: float indicating the number of batches per epoch.

  Returns:
    A scalar float tensor, representing the learning rate.

  Raises:
    ValueError: piecewise_learning_rate_schedule is not formatted correctly.
  """
  pieces = piecewise_learning_rate_schedule.split(';')
  if len(pieces) % 2 == 0:
    raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                     'number of components')
  values = []
  boundaries = []
  for i, piece in enumerate(pieces):
    if i % 2 == 0:
      try:
        values.append(float(piece))
      except ValueError:
        raise ValueError('Invalid learning rate: ' + piece)
    else:
      try:
        boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
      except ValueError:
        raise ValueError('Invalid epoch: ' + piece)
  return tf.train.piecewise_constant(global_step, boundaries, values,
                                     name='piecewise_learning_rate')


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)

  if params.piecewise_learning_rate_schedule:
    if (params.learning_rate or params.learning_rate_decay_factor or
        params.minimum_learning_rate or params.num_epochs_per_decay):
      raise ValueError('No other learning rate-related flags can be specified '
                       'if --piecewise_learning_rate_schedule is specified')
    return get_piecewise_learning_rate(params.piecewise_learning_rate_schedule,
                                       global_step, num_batches_per_epoch)

  learning_rate = (
      params.learning_rate or model.get_learning_rate(global_step, batch_size))
  if (params.learning_rate and params.num_epochs_per_decay > 0 and
      params.learning_rate_decay_factor > 0):
    decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        params.learning_rate,
        global_step,
        decay_steps,
        params.learning_rate_decay_factor,
        staircase=True)

    if params.minimum_learning_rate != 0.:
      learning_rate = tf.maximum(learning_rate,
                                 params.minimum_learning_rate)
  return learning_rate


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params
    self.dataset = datasets.create_dataset(self.params.data_dir,
                                           self.params.data_name)
    self.model = model_config.get_model_config(self.params.model, self.dataset)
    self.trace_filename = self.params.trace_file
    self.data_format = self.params.data_format
    self.enable_layout_optimizer = self.params.enable_layout_optimizer
    autotune_threshold = self.params.autotune_threshold if (
        self.params.autotune_threshold) else 1
    min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
    self.num_warmup_batches = self.params.num_warmup_batches if (
        self.params.num_warmup_batches is not None) else max(
            10, min_autotune_warmup)
    self.graph_file = self.params.graph_file
    self.resize_method = self.params.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = self.params.num_gpus
    if self.params.gpu_indices:
      self.gpu_indices = [int(x) for x in self.params.gpu_indices.split(',')]
    else:
      self.gpu_indices = [x for x in range(self.num_gpus)]
    self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()

    if (self.params.device == 'cpu' and self.params.data_format == 'NCHW' and
        not self.params.mkl):
      raise ValueError('device=cpu requires that data_format=NHWC')

    if ((self.params.num_epochs_per_decay or
         self.params.learning_rate_decay_factor) and
        not (self.params.learning_rate and self.params.num_epochs_per_decay and
             self.params.learning_rate_decay_factor)):
      raise ValueError('If one of num_epochs_per_decay or '
                       'learning_rate_decay_factor is set, both must be set'
                       'and learning_rate must be set')
    if (self.params.minimum_learning_rate and
        not (self.params.learning_rate and self.params.num_epochs_per_decay and
             self.params.learning_rate_decay_factor)):
      raise ValueError('minimum_learning_rate requires learning_rate,'
                       'num_epochs_per_decay, and '
                       'learning_rate_decay_factor to be set')

    if (self.params.use_fp16 and self.params.fp16_vars and
        'replicated' in self.params.variable_update and
        self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec):
      raise ValueError('fp16 variables are not supported with NCCL')

    if self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale:
      if self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec:
        raise ValueError('Automatic loss scaling is not supported with NCCL.')
      if self.params.variable_update not in ('parameter_server', 'replicated'):
        raise ValueError('Automatic loss scaling is not supported with '
                         'variable_update=%s.' % self.params.variable_update)
      if self.params.staged_vars:
        raise ValueError('Automatic loss scaling is not supported with'
                         'staged_vars.')

    if (self.params.debugger is not None and self.params.debugger != 'cli' and
        ':' not in self.params.debugger):
      raise ValueError('--debugger must be "cli" or in the form '
                       'host:port')

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if self.params.batch_size > 0:
      self.model.set_batch_size(self.params.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus
    self.batch_group_size = self.params.batch_group_size
    self.enable_auto_loss_scale = (
        self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
    self.loss_scale = None
    self.loss_scale_normal_steps = None

    self.job_name = self.params.job_name  # "" for local training

    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and (self.params.variable_update !=
                                       'distributed_all_reduce')
    # controller is used for distributed_all_reduce with > 1 worker.
    use_controller = (
        self.params.variable_update == 'distributed_all_reduce' and
        self.job_name)
    if use_controller and not params.controller_host:
      raise ValueError('When variable_update==distributed_all_reduce '
                       'controller_host must also be specified.')

    self.local_parameter_device_flag = self.params.local_parameter_device
    if self.job_name:
      self.task_index = self.params.task_index
      self.cluster_manager = platforms_util.get_cluster_manager(
          params, create_config_proto(params))
      assert isinstance(self.cluster_manager, cnn_util.BaseClusterManager)

      worker_prefix = '/job:worker/task:%s' % self.task_index
      if use_ps_server:
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0',
            cluster=self.cluster_manager.get_cluster_spec())
        # This device on which the queues for managing synchronization between
        # servers should be stored.
        self.sync_queue_devices = [
            '/job:ps/task:%s/cpu:0' % i
            for i in range(self.cluster_manager.num_ps())
        ]
      else:
        self.sync_queue_devices = ['/job:worker/task:0/cpu:0']
    else:
      self.task_index = 0
      self.cluster_manager = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % self.params.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]

    self.num_workers = (
        self.cluster_manager.num_workers() if self.cluster_manager else 1)
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0

    if self.num_workers > 1 and self.params.all_reduce_spec == 'nccl':
      raise ValueError('--all_reduce_spec=nccl is invalid in a '
                       'multi-worker job')

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]

    subset = 'validation' if params.eval else 'train'
    self.num_batches, self.num_epochs = get_num_batches_and_epochs(
        params, self.batch_size * self.num_workers,
        self.dataset.num_examples_per_epoch(subset))

    if (self.params.staged_vars and
        self.params.variable_update != 'parameter_server'):
      raise ValueError('staged_vars for now is only supported with '
                       'variable_update=parameter_server')

    if self.params.variable_update == 'parameter_server':
      if self.job_name:
        if not self.params.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
        else:
          self.variable_mgr = (
              variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
      else:
        if not self.params.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
              self)
    elif self.params.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec,
          self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group)
    elif self.params.variable_update == 'distributed_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
          self, self.params.all_reduce_spec,
          ('worker' if self.num_workers > 1 else 'localhost'), self.num_workers,
          self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group)
    elif self.params.variable_update == 'distributed_replicated':
      assert self.params.cross_replica_sync
      if not self.job_name:
        raise ValueError('Invalid variable_update in local mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif self.params.variable_update == 'independent':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError(
          'Invalid variable_update: %s' % self.params.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      if use_ps_server:
        self.global_step_device = self.param_server_device
      else:
        self.global_step_device = '/job:worker/task:0/cpu:0'
    else:
      self.global_step_device = self.cpu_device

    self.image_preprocessor = self.get_image_preprocessor()
    self.init_global_step = 0

  def reset_devices_for_task(self, task_num, is_local=False):
    """Used to imitate another task when building a distributed graph."""
    worker_prefix = ('job:localhost'
                     if is_local else '/job:worker/task:%s' % task_num)
    self.cpu_device = '%s/cpu:0' % worker_prefix
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]
    self.devices = self.variable_mgr.get_devices()

  def raw_devices_across_tasks(self, is_local=False):
    """Returns list of raw device names across all tasks."""
    if is_local:
      assert self.num_workers == 1
      return self.raw_devices
    else:
      return [
          'job:worker/task%s/%s:%i' % (t, self.params.device, i)
          for t in xrange(self.num_workers)
          for i in xrange(self.num_gpus)
      ]

  def print_info(self):
    """Print basic information."""
    log_fn('Model:       %s' % self.model.get_model())
    dataset_name = self.dataset.name
    if self.dataset.use_synthetic_gpu_images():
      dataset_name += ' (synthetic)'
    log_fn('Dataset:     %s' % dataset_name)
    log_fn('Mode:        %s' % get_mode_from_params(self.params))
    single_session = self.params.variable_update == 'distributed_all_reduce'
    log_fn('SingleSess:  %s' % single_session)
    if single_session:
      device_list = self.raw_devices_across_tasks()
    else:
      device_list = self.raw_devices
    log_fn('Batch size:  %s global' % (self.batch_size * self.num_workers))
    log_fn('             %s per device' % (self.batch_size /
                                           len(self.raw_devices)))
    if self.batch_group_size > 1:
      log_fn('             %d batches per prepocessing group' %
             self.batch_group_size)
    log_fn('Num batches: %d' % self.num_batches)
    log_fn('Num epochs:  %.2f' % self.num_epochs)
    log_fn('Devices:     %s' % device_list)
    log_fn('Data format: %s' % self.data_format)
    log_fn('Layout optimizer: %s' % self.enable_layout_optimizer)
    log_fn('Optimizer:   %s' % self.params.optimizer)
    log_fn('Variables:   %s' % self.params.variable_update)
    if (self.params.variable_update == 'replicated' or
        self.params.variable_update == 'distributed_all_reduce'):
      log_fn('AllReduce:   %s' % self.params.all_reduce_spec)
    if self.job_name:
      log_fn('Sync:        %s' % self.params.cross_replica_sync)
    if self.params.staged_vars:
      log_fn('Staged vars: %s' % self.params.staged_vars)
    log_fn('==========')

  def run(self):
    """Run the benchmark task assigned to this process.

    Returns:
      Dictionary of statistics for training or eval.
    Raises:
       ValueError: unrecognized job name.
    """
    if self.params.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.cluster_manager.join_server()
      return {}

    # For distributed_all_reduce with multiple workers, drive
    # from a separate controller process.
    if self.params.variable_update == 'distributed_all_reduce':
      if self.params.job_name == 'worker':
        log_fn('Starting worker %s' % self.task_index)
        self.cluster_manager.join_server()
        return
      elif self.params.job_name and self.params.job_name != 'controller':
        raise ValueError('unrecognized job name: %s' % self.params.job_name)

    with tf.Graph().as_default():
      if self.params.eval:
        return self._eval_cnn()
      else:
        return self._benchmark_cnn()

  def _eval_cnn(self):
    """Evaluate a model every self.params.eval_interval_secs.

    Returns:
      Dictionary containing eval statistics. Currently returns an empty
      dictionary.
    """
    (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    saver = tf.train.Saver(self.variable_mgr.savable_variables())
    summary_writer = tf.summary.FileWriter(self.params.eval_dir,
                                           tf.get_default_graph())
    target = ''
    local_var_init_op = tf.local_variables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    summary_op = tf.summary.merge_all()
    # TODO(huangyp): Check if checkpoints haven't updated for hours and abort.
    while True:
      self._eval_once(saver, summary_writer, target, local_var_init_op_group,
                      image_producer_ops, enqueue_ops, fetches, summary_op)
      if self.params.eval_interval_secs <= 0:
        break
      time.sleep(self.params.eval_interval_secs)
    return {}

  def _eval_once(self, saver, summary_writer, target, local_var_init_op_group,
                 image_producer_ops, enqueue_ops, fetches, summary_op):
    """Evaluate the model from a checkpoint using validation dataset."""
    with tf.Session(
        target=target, config=create_config_proto(self.params)) as sess:
      if self.params.train_dir is None:
        raise ValueError('Trained model directory not specified')
      try:
        global_step = load_checkpoint(saver, sess, self.params.train_dir)
      except CheckpointNotFoundException:
        log_fn('Checkpoint not found in %s' % self.params.train_dir)
        return
      sess.run(local_var_init_op_group)
      if self.dataset.queue_runner_required():
        tf.train.start_queue_runners(sess=sess)
      image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                              self.batch_group_size)
      image_producer.start()
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i + 1)])
        image_producer.notify_image_consumption()
      loop_start_time = start_time = time.time()
      top_1_accuracy_sum = 0.0
      top_5_accuracy_sum = 0.0
      total_eval_count = self.num_batches * self.batch_size
      for step in xrange(self.num_batches):
        if (self.params.save_summaries_steps > 0 and
            (step + 1) % self.params.save_summaries_steps == 0):
          results, summary_str = sess.run([fetches, summary_op])
          summary_writer.add_summary(summary_str)
        else:
          results = sess.run(fetches)
        top_1_accuracy_sum += results['top_1_accuracy']
        top_5_accuracy_sum += results['top_5_accuracy']
        if (step + 1) % self.params.display_every == 0:
          duration = time.time() - start_time
          examples_per_sec = (
              self.batch_size * self.params.display_every / duration)
          log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
          start_time = time.time()
        image_producer.notify_image_consumption()
      loop_end_time = time.time()
      image_producer.done()
      accuracy_at_1 = top_1_accuracy_sum / self.num_batches
      accuracy_at_5 = top_5_accuracy_sum / self.num_batches
      summary = tf.Summary()
      summary.value.add(tag='eval/Accuracy@1', simple_value=accuracy_at_1)
      summary.value.add(tag='eval/Accuracy@5', simple_value=accuracy_at_5)
      summary_writer.add_summary(summary, global_step)
      log_fn('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f [%d examples]' %
             (accuracy_at_1, accuracy_at_5, total_eval_count))
      elapsed_time = loop_end_time - loop_start_time
      images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
      # Note that we compute the top 1 accuracy and top 5 accuracy for each
      # batch, which will have a slight performance impact.
      log_fn('-' * 64)
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)

  def _benchmark_cnn(self):
    """Run cnn in benchmark mode. Skip the backward pass if forward_only is on.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    if self.params.variable_update == 'distributed_all_reduce':
      self.single_session = True
      (image_producer_ops, enqueue_ops, fetches) = (
          self._build_model_single_session())
    else:
      self.single_session = False
      (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list)
    execution_barrier = None
    if (not self.single_session and self.job_name and
        not self.params.cross_replica_sync):
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)

    if ((not self.single_session) and self.job_name and
        self.params.cross_replica_sync):
      # Block all replicas until all replicas are ready for next step.
      fetches['sync_queues'] = self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group])

    local_var_init_op = tf.local_variables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)

    summary_op = tf.summary.merge_all()
    is_chief = (not self.job_name or self.task_index == 0)
    summary_writer = None
    if (is_chief and self.params.summary_verbosity and self.params.train_dir and
        self.params.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(self.params.train_dir,
                                             tf.get_default_graph())

    # We want to start the benchmark timer right after a image_producer barrier
    # and avoids undesired waiting times on barriers.
    if ((self.num_warmup_batches + len(enqueue_ops) - 1) %
        self.batch_group_size) != 0:
      self.num_warmup_batches = int(
          math.ceil((self.num_warmup_batches + len(enqueue_ops) - 1.0) /
                    (self.batch_group_size
                    )) * self.batch_group_size - len(enqueue_ops) + 1)
      log_fn('Round up warm up steps to %d to match batch_group_size' %
             self.num_warmup_batches)
      assert ((self.num_warmup_batches + len(enqueue_ops) - 1) %
              self.batch_group_size) == 0
    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    saver = tf.train.Saver(
        self.variable_mgr.savable_variables(), save_relative_paths=True)
    ready_for_local_init_op = None
    if self.job_name and not self.single_session:
      # In distributed mode, we don't want to run local_var_init_op_group until
      # the global variables are initialized, because local_var_init_op_group
      # may use global variables (such as in distributed replicated mode). We
      # don't set this in non-distributed mode, because in non-distributed mode,
      # local_var_init_op_group may itself initialize global variables (such as
      # in replicated mode).
      ready_for_local_init_op = tf.report_uninitialized_variables(
          tf.global_variables())
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=self.params.train_dir,
        ready_for_local_init_op=ready_for_local_init_op,
        local_init_op=local_var_init_op_group,
        saver=saver,
        global_step=global_step,
        summary_op=None,
        save_model_secs=self.params.save_model_secs,
        summary_writer=summary_writer)

    step_train_times = []
    start_standard_services = (
        self.params.summary_verbosity >= 1 or
        self.dataset.queue_runner_required())
    target = self.cluster_manager.get_target() if self.cluster_manager else ''
    with sv.managed_session(
        master=target,
        config=create_config_proto(self.params),
        start_standard_services=start_standard_services) as sess:
      image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                              self.batch_group_size)
      image_producer.start()
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i + 1)])
        image_producer.notify_image_consumption()
      self.init_global_step, = sess.run([global_step])
      if not self.single_session:
        global_step_watcher = GlobalStepWatcher(
            sess, global_step,
            self.num_workers * self.num_warmup_batches +
            self.init_global_step,
            self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
        global_step_watcher.start()
      else:
        global_step_watcher = None

      if self.graph_file is not None:
        path, filename = os.path.split(self.graph_file)
        as_text = filename.endswith('txt')
        log_fn('Writing GraphDef as %s to %s' % (  # pyformat break
            'text' if as_text else 'binary', self.graph_file))
        tf.train.write_graph(sess.graph_def, path, filename, as_text)

      log_fn('Running warm up')
      local_step = -1 * self.num_warmup_batches

      if self.single_session or (self.params.cross_replica_sync and
                                 self.params.job_name):
        # In cross-replica sync mode, all workers must run the same number of
        # local steps, or else the workers running the extra step will block.
        done_fn = lambda: local_step == self.num_batches
      else:
        done_fn = global_step_watcher.done
      if self.params.debugger is not None:
        if self.params.debugger == 'cli':
          log_fn('The CLI TensorFlow debugger will be used.')
          sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        else:
          log_fn('The TensorBoard debugger plugin will be used.')
          sess = tf_debug.TensorBoardDebugWrapperSession(sess,
                                                         self.params.debugger)
      loop_start_time = time.time()
      while not done_fn():
        if local_step == 0:
          log_fn('Done warm up')
          if execution_barrier:
            log_fn('Waiting for other replicas to finish warm up')
            assert global_step_watcher.start_time == 0
            sess.run([execution_barrier])

          header_str = 'Step\tImg/sec\tloss'
          if self.params.print_training_accuracy or self.params.forward_only:
            header_str += '\ttop_1_accuracy\ttop_5_accuracy'
          log_fn(header_str)
          assert len(step_train_times) == self.num_warmup_batches
          # reset times to ignore warm up batch
          step_train_times = []
          loop_start_time = time.time()
        if (summary_writer and
            (local_step + 1) % self.params.save_summaries_steps == 0):
          fetch_summary = summary_op
        else:
          fetch_summary = None
        summary_str = benchmark_one_step(
            sess, fetches, local_step,
            self.batch_size * (self.num_workers if self.single_session else 1),
            step_train_times, self.trace_filename, image_producer, self.params,
            fetch_summary)
        if summary_str is not None and is_chief:
          sv.summary_computed(sess, summary_str)
        local_step += 1
      loop_end_time = time.time()
      # Waits for the global step to be done, regardless of done_fn.
      if global_step_watcher:
        while not global_step_watcher.done():
          time.sleep(.25)
      if self.single_session:
        elapsed_time = loop_end_time - loop_start_time
        average_wall_time = elapsed_time / local_step if local_step > 0 else 0
        images_per_sec = (self.num_workers * local_step * self.batch_size /
                          elapsed_time)
        num_steps = local_step * self.num_workers
      else:
        # NOTE: Each worker independently increases the global step. So,
        # num_steps will be the sum of the local_steps from each worker.
        num_steps = global_step_watcher.num_steps()
        elapsed_time = global_step_watcher.elapsed_time()
        average_wall_time = (elapsed_time * self.num_workers / num_steps
                             if num_steps > 0 else 0)
        images_per_sec = num_steps * self.batch_size / elapsed_time

      log_fn('-' * 64)
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)
      image_producer.done()
      if is_chief:
        store_benchmarks({'total_images_per_sec': images_per_sec}, self.params)
      # Save the model checkpoint.
      if self.params.train_dir is not None and is_chief:
        checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
        if not gfile.Exists(self.params.train_dir):
          gfile.MakeDirs(self.params.train_dir)
        sv.saver.save(sess, checkpoint_path, global_step)

      if execution_barrier:
        # Wait for other workers to reach the end, so this worker doesn't
        # go away underneath them.
        sess.run([execution_barrier])
    sv.stop()
    return {
        'num_workers': self.num_workers,
        'num_steps': num_steps,
        'average_wall_time': average_wall_time,
        'images_per_sec': images_per_sec
    }

  def _build_image_processing(self, shift_ratio=0):
    """"Build the image (pre)processing portion of the model graph."""
    with tf.device(self.cpu_device):
      if self.params.eval:
        subset = 'validation'
      else:
        subset = 'train'
      image_producer_ops = []
      image_producer_stages = []
      images_splits, labels_splits = self.image_preprocessor.minibatch(
          self.dataset,
          subset=subset,
          use_datasets=self.params.use_datasets,
          cache_data=self.params.cache_data,
          shift_ratio=shift_ratio)
      images_shape = images_splits[0].get_shape()
      labels_shape = labels_splits[0].get_shape()
      for device_num in range(len(self.devices)):
        image_producer_stages.append(
            data_flow_ops.StagingArea(
                [images_splits[0].dtype, labels_splits[0].dtype],
                shapes=[images_shape, labels_shape]))
        for group_index in xrange(self.batch_group_size):
          if not self.use_synthetic_gpu_images:
            batch_index = group_index + device_num * self.batch_group_size
            put_op = image_producer_stages[device_num].put(
                [images_splits[batch_index], labels_splits[batch_index]])
            image_producer_ops.append(put_op)
    return (image_producer_ops, image_producer_stages)

  def _build_model(self):
    """Build the TensorFlow graph."""
    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    phase_train = not (self.params.eval or self.params.forward_only)

    log_fn('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    enqueue_ops = []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()
      if self.params.use_fp16:
        init_loss_scale_val = float(self.params.fp16_loss_scale or
                                    self.model.get_fp16_loss_scale())
        if self.enable_auto_loss_scale or init_loss_scale_val != 1:
          self.loss_scale = tf.get_variable(
              name='loss_scale',
              initializer=init_loss_scale_val,
              dtype=tf.float32,
              trainable=False)
          self.loss_scale_normal_steps = tf.get_variable(
              name='loss_scale_normal_steps', initializer=0, trainable=False)
        else:
          self.loss_scale = None
          self.loss_scale_normal_steps = None

    # Build the processing and model for the worker.
    (image_producer_ops,
     image_producer_stages) = self._build_image_processing(shift_ratio=0)
    image_producer_ops = tf.group(*image_producer_ops)
    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with self.variable_mgr.create_outer_variable_scope(
          device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, device_num,
            image_producer_stages[device_num], gpu_compute_stage_ops,
            gpu_grad_stage_ops)
        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        if not phase_train or self.params.print_training_accuracy:
          all_top_1_ops.append(results['top_1_op'])
          all_top_5_ops.append(results['top_5_op'])

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

    if self.variable_mgr.supports_staged_vars():
      for staging_ops in self.variable_mgr.staging_vars_on_devices:
        gpu_compute_stage_ops.extend(
            [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
    enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
    if gpu_grad_stage_ops:
      staging_delta_ops += gpu_grad_stage_ops
    if staging_delta_ops:
      enqueue_ops.append(tf.group(*(staging_delta_ops)))

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_top_1_ops,
                                  all_top_5_ops, phase_train)
    return (image_producer_ops, enqueue_ops, fetches)

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops, all_top_1_ops, all_top_5_ops,
                     phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {'enqueue_ops': enqueue_ops}
    if all_top_1_ops:
      fetches['top_1_accuracy'] = tf.reduce_sum(all_top_1_ops) / self.batch_size
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('top_1_accuracy', fetches['top_1_accuracy'])
    if all_top_5_ops:
      fetches['top_5_accuracy'] = tf.reduce_sum(all_top_5_ops) / self.batch_size
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('top_5_accuracy', fetches['top_5_accuracy'])

    if not phase_train:
      if self.params.forward_only:
        fetches['all_logits'] = tf.concat(all_logits, 0)
      return fetches
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        total_loss = tf.reduce_mean(losses)
        avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

        gradient_clip = self.params.gradient_clip
        learning_rate = get_learning_rate(self.params, global_step,
                                          self.dataset.num_examples_per_epoch(),
                                          self.model, self.batch_size)

        if gradient_clip is not None:
          clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                             +gradient_clip), var)
                           for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        if self.params.optimizer == 'momentum':
          opt = tf.train.MomentumOptimizer(
              learning_rate, self.params.momentum, use_nesterov=True)
        elif self.params.optimizer == 'sgd':
          opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.params.optimizer == 'rmsprop':
          opt = tf.train.RMSPropOptimizer(
              learning_rate,
              self.params.rmsprop_decay,
              momentum=self.params.rmsprop_momentum,
              epsilon=self.params.rmsprop_epsilon)
        else:
          raise ValueError('Optimizer "%s" was not recognized',
                           self.params.optimizer)

        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=self.loss_scale,
            loss_scale_normal_steps=self.loss_scale_normal_steps,
            inc_loss_scale_every_n=self.params.fp16_inc_loss_scale_every_n,
            is_chief=not self.job_name or self.task_index == 0)

        self.variable_mgr.append_apply_gradients_ops(
            gradient_state, opt, clipped_grads, training_ops, loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops))

    with tf.device(self.cpu_device):
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        if self.loss_scale is not None:
          tf.summary.scalar('loss_scale', self.loss_scale)
          tf.summary.scalar('loss_scale_normal_steps',
                            self.loss_scale_normal_steps)

        if self.params.summary_verbosity >= 2:
          # Histogram of log values of all non-zero gradients.
          all_grads = []
          for grad, var in avg_grads:
            all_grads.append(tf.reshape(grad, [-1]))
          grads = tf.abs(tf.concat(all_grads, 0))
          # exclude grads with zero values.
          indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
          log_grads = tf.reshape(
              tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
          tf.summary.histogram('log_gradients', log_grads)

        if self.params.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    fetches['train_op'] = train_op
    fetches['total_loss'] = total_loss
    return fetches

  def _build_model_single_session(self):
    """Build the TensorFlow graph for multiple replicas in a single_session.

    Returns:
      image_producer_ops:
      enqueue_ops:
      fetches:

    Raises:
       ValueError: optimizer not recognized.

    Single session runs multiple model replicas as part of one large
    distributed graph, whose global execution is always step-synchronized.
    """
    # verify assumptions
    assert self.params.task_index == 0
    assert not self.params.eval
    assert not self.params.forward_only
    assert not self.params.staged_vars

    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    phase_train = True

    log_fn('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    enqueue_ops = []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    update_ops = []
    global_image_producer_ops = []

    is_local = not self.job_name
    if is_local:
      assert self.num_workers == 1
    for task_num in range(self.num_workers):
      # Reset the devices that self.variable_mgr knows about to those
      # belonging to the next worker (task).
      self.reset_devices_for_task(task_num, is_local)
      # Build the per-worker image processing
      (image_producer_ops, image_producer_stages) = (
          self._build_image_processing(
              shift_ratio=(float(task_num) / self.num_workers)))
      global_image_producer_ops.extend(image_producer_ops)
      # Build the per-worker model replica.
      for rel_device_num in range(len(self.devices)):
        abs_device_num = task_num * len(self.devices) + rel_device_num
        with self.variable_mgr.create_outer_variable_scope(
            abs_device_num), tf.name_scope(
                'task_%i_tower_%i' % (task_num, rel_device_num)) as name_scope:
          task_results = self.add_forward_pass_and_gradients(
              phase_train, rel_device_num, abs_device_num,
              image_producer_stages[rel_device_num], gpu_compute_stage_ops,
              gpu_grad_stage_ops)
          if phase_train:
            losses.append(task_results['loss'])
            device_grads.append(task_results['gradvars'])
          else:
            all_logits.append(task_results['logits'])
          if not phase_train or self.params.print_training_accuracy:
            all_top_1_ops.append(task_results['top_1_op'])
            all_top_5_ops.append(task_results['top_5_op'])

          if rel_device_num == 0:
            # Retain the Batch Normalization updates operations only
            # from the first tower. These operations update the moving
            # mean and moving variance variables, which are updated
            # (but not used) during training, and used during
            # evaluation. The moving mean and variance approximate the
            # true mean and variance across all images in the
            # dataset. Therefore, in replicated mode, these moving
            # averages would be almost identical for each tower, and
            # so we only update and save the moving averages for one
            # tower. In parameter server mode, all towers share a copy
            # of the variables so we also only need to update and save
            # the moving averages once.
            update_ops.extend(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
            assert not self.variable_mgr.staging_delta_ops

    enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
    assert not self.variable_mgr.supports_staged_vars()
    assert not gpu_grad_stage_ops

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_top_1_ops,
                                  all_top_5_ops, phase_train)
    global_image_producer_ops = tf.group(*global_image_producer_ops)
    return (global_image_producer_ops, enqueue_ops, fetches)

  def add_forward_pass_and_gradients(self, phase_train, rel_device_num,
                                     abs_device_num, image_producer_stage,
                                     gpu_compute_stage_ops, gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes + 1
    input_data_type = get_data_type(self.params)
    data_type = get_data_type(self.params)
    if not self.use_synthetic_gpu_images:
      with tf.device(self.cpu_device):
        host_images, host_labels = image_producer_stage.get()
        images_shape = host_images.get_shape()
        labels_shape = host_labels.get_shape()
    with tf.device(self.raw_devices[rel_device_num]):
      if not self.use_synthetic_gpu_images:
        gpu_compute_stage = data_flow_ops.StagingArea(
            [host_images.dtype, host_labels.dtype],
            shapes=[images_shape, labels_shape])
        # The CPU-to-GPU copy is triggered here.
        gpu_compute_stage_op = gpu_compute_stage.put([host_images, host_labels])
        images, labels = gpu_compute_stage.get()
        images = tf.reshape(images, shape=images_shape)
        gpu_compute_stage_ops.append(gpu_compute_stage_op)
      else:
        # Minor hack to avoid H2D copy when using synthetic data
        image_size = self.model.get_image_size()
        image_shape = [
            self.batch_size // self.num_gpus, image_size, image_size,
            self.dataset.depth
        ]
        labels_shape = [self.batch_size // self.num_gpus]
        # Synthetic image should be within [0, 255].
        images = tf.truncated_normal(
            image_shape,
            dtype=input_data_type,
            mean=127,
            stddev=60,
            name='synthetic_images')
        images = tf.contrib.framework.local_variable(
            images, name='gpu_cached_images')
        labels = tf.random_uniform(
            labels_shape,
            minval=0,
            maxval=nclass - 1,
            dtype=tf.int32,
            name='synthetic_labels')

    with tf.device(self.devices[rel_device_num]):
      # Rescale from [0, 255] to [0, 2]
      images = tf.multiply(images, 1. / 127.5)
      # Rescale to [-1, 1]
      images = tf.subtract(images, 1.0)

      if self.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
      if input_data_type != data_type:
        images = tf.cast(images, data_type)
      var_type = tf.float32
      if data_type == tf.float16 and self.params.fp16_vars:
        var_type = tf.float16
      network = convnet_builder.ConvNetBuilder(
          images, self.dataset.depth, phase_train, self.params.use_tf_layers,
          self.data_format, data_type, var_type)
      with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
        self.model.add_inference(network)
        # Add the final fully-connected class layer
        logits = network.affine(nclass, activation='linear')
        aux_logits = None
        if network.aux_top_layer is not None:
          with network.switch_to_aux_top_layer():
            aux_logits = network.affine(
                nclass, activation='linear', stddev=0.001)
      if data_type == tf.float16:
        # TODO(reedwm): Determine if we should do this cast here.
        logits = tf.cast(logits, tf.float32)
        if aux_logits is not None:
          aux_logits = tf.cast(aux_logits, tf.float32)

      results = {}  # The return value
      if not phase_train or self.params.print_training_accuracy:
        top_1_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
        top_5_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
        results['top_1_op'] = top_1_op
        results['top_5_op'] = top_5_op

      if not phase_train:
        results['logits'] = logits
        return results
      loss = loss_function(logits, labels, aux_logits=aux_logits)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      if data_type == tf.float16 and self.params.fp16_vars:
        # fp16 reductions are very slow on GPUs, so cast to fp32 before calling
        # tf.nn.l2_loss and tf.add_n.
        # TODO(b/36217816): Once the bug is fixed, investigate if we should do
        # this reduction in fp16.
        fp32_params = (tf.cast(p, tf.float32) for p in params)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
      else:
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
      weight_decay = self.params.weight_decay
      if weight_decay is not None and weight_decay != 0.:
        loss += weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = loss if self.loss_scale is None else loss * self.loss_scale
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      if self.loss_scale is not None:
        # TODO(reedwm): If automatic loss scaling is not used, we could avoid
        # these multiplications by directly modifying the learning rate instead.
        # If this is done, care must be taken to ensure that this scaling method
        # is correct, as some optimizers square gradients and do other
        # operations which might not be compatible with modifying both the
        # gradients and the learning rate.

        grads = [
            grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
        ]

      if self.params.staged_vars:
        grad_dtypes = [grad.dtype for grad in grads]
        grad_shapes = [grad.shape for grad in grads]
        grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
        grad_stage_op = grad_stage.put(grads)
        # In general, this decouples the computation of the gradients and
        # the updates of the weights.
        # During the pipeline warm up, this runs enough training to produce
        # the first set of gradients.
        gpu_grad_stage_ops.append(grad_stage_op)
        grads = grad_stage.get()

      param_refs = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num, writable=True)
      gradvars = list(zip(grads, param_refs))
      results['loss'] = loss
      results['gradvars'] = gradvars
      return results

  def get_image_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    image_size = self.model.get_image_size()
    input_data_type = get_data_type(self.params)

    shift_ratio = 0
    if self.job_name:
      # shift_ratio prevents multiple workers from processing the same batch
      # during a step
      shift_ratio = float(self.task_index) / self.num_workers

    processor_class = self.dataset.get_image_preprocessor()
    assert processor_class
    return processor_class(
        image_size,
        image_size,
        self.batch_size * self.batch_group_size,
        len(self.devices) * self.batch_group_size,
        dtype=input_data_type,
        train=(not self.params.eval),
        distortions=self.params.distortions,
        resize_method=self.resize_method,
        shift_ratio=shift_ratio,
        summary_verbosity=self.params.summary_verbosity,
        distort_color_in_yiq=self.params.distort_color_in_yiq,
        fuse_decode_and_crop=self.params.fuse_decode_and_crop)

  def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      an op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    with tf.device(self.sync_queue_devices[(
        self.sync_queue_counter % len(self.sync_queue_devices))]):
      sync_queues = [
          tf.FIFOQueue(self.num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(self.num_workers)]
      queue_ops = []
      # For each other worker, add an entry in a queue, signaling that it can
      # finish this step.
      token = tf.constant(False)
      with tf.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == self.task_index:
            queue_ops.append(tf.no_op())
          else:
            queue_ops.append(q.enqueue(token))

      # Drain tokens off queue for this worker, one for each other worker.
      queue_ops.append(
          sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

      return tf.group(*queue_ops)


def store_benchmarks(names_to_values, params):
  if params.result_storage:
    benchmark_storage.store_benchmark(names_to_values, params.result_storage)


def setup(params):
  """Sets up the environment that BenchmarkCNN should run in.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  if params.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  if params.autotune_threshold:
    os.environ['TF_AUTOTUNE_THRESHOLD'] = str(params.autotune_threshold)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets environment variables for MKL
  if params.mkl:
    os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
    os.environ['KMP_SETTINGS'] = str(params.kmp_settings)
    os.environ['KMP_AFFINITY'] = params.kmp_affinity
    if params.num_intra_threads > 0:
      os.environ['OMP_NUM_THREADS'] = str(params.num_intra_threads)

  # Sets GPU thread settings
  params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
  if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
    raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
  os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

  if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
    raise ValueError(
        'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
        params.per_gpu_thread_count)
  # Default to two threads. One for the device compute and the other for
  # memory copies.
  per_gpu_thread_count = params.per_gpu_thread_count or 2
  total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

  if params.gpu_thread_mode == 'gpu_private':
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  elif params.gpu_thread_mode == 'gpu_shared':
    os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

  if not params.num_inter_threads and params.gpu_thread_mode in [
      'gpu_private', 'gpu_shared'
  ]:
    cpu_count = multiprocessing.cpu_count()
    main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
    params = params._replace(num_inter_threads=main_thread_count)

  platforms_util.initialize(params, create_config_proto(params))

  return params
