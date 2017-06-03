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

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import os
import threading
import time

import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
import benchmark_storage
import cnn_util
import datasets
import model_config
import preprocessing
import variable_mgr

tf.flags.DEFINE_string('model', 'trivial', 'name of the model to run')

# The code will first check if it's running under benchmarking mode
# or evaluation mode, depending on FLAGS.eval:
# Under the evaluation mode, this script will read a saved model,
#   and compute the accuracy of the model against a validation dataset.
#   Additional ops for accuracy and top_k predictors are only used under this
#   mode.
# Under the benchmarking mode, user can specify whether nor not to use
#   the forward-only option, which will only compute the loss function.
#   forward-only cannot be enabled with eval at the same time.
tf.flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')
tf.flags.DEFINE_boolean('forward_only', False, """whether use forward-only or
                         training for benchmarking""")
tf.flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
tf.flags.DEFINE_integer('num_batches', 100,
                        'number of batches to run, excluding warmup')
tf.flags.DEFINE_integer('num_warmup_batches', None,
                        'number of batches to run before timing')
tf.flags.DEFINE_integer('autotune_threshold', None,
                        'The autotune threshold for the models')
tf.flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
tf.flags.DEFINE_integer('display_every', 10,
                        """Number of local steps after which progress is printed
                        out""")
tf.flags.DEFINE_string('data_dir', None, """Path to dataset in TFRecord format
                       (aka Example protobufs). If not specified,
                       synthetic data will be used.""")
tf.flags.DEFINE_string('data_name', None,
                       """Name of dataset: imagenet or flowers.
                       If not specified, it is automatically guessed
                       based on --data_dir.""")
tf.flags.DEFINE_string('resize_method', 'bilinear',
                       """Method for resizing input images:
                       crop,nearest,bilinear,bicubic or area.
                       The 'crop' mode requires source images to be at least
                       as large as the network input size,
                       while the other modes support any sizes and apply
                       random bbox distortions
                       before resizing (even with --nodistortions).""")
tf.flags.DEFINE_boolean('distortions', True,
                        """Enable/disable distortions during
                       image preprocessing. These include bbox and color
                       distortions.""")
tf.flags.DEFINE_string('local_parameter_device', 'gpu',
                       """Device to use as parameter server: cpu or gpu.
                          For distributed training, it can affect where caching
                          of variables happens.""")
tf.flags.DEFINE_string('device', 'gpu',
                       """Device to use for computation: cpu or gpu""")
tf.flags.DEFINE_string('data_format', 'NCHW',
                       """Data layout to use: NHWC (TF native)
                       or NCHW (cuDNN native).""")
tf.flags.DEFINE_integer('num_intra_threads', 1,
                        """Number of threads to use for intra-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")
tf.flags.DEFINE_integer('num_inter_threads', 0,
                        """Number of threads to use for inter-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")
tf.flags.DEFINE_string('trace_file', None,
                       """Enable TensorFlow tracing and write trace to
                       this file.""")
tf.flags.DEFINE_string('graph_file', None,
                       """Write the model's graph definition to this
                       file. Defaults to binary format unless filename ends
                       in 'txt'.""")
tf.flags.DEFINE_string('optimizer', 'sgd',
                       'Optimizer to use: momentum or sgd or rmsprop')
tf.flags.DEFINE_float('learning_rate', None,
                      """Initial learning rate for training.""")
tf.flags.DEFINE_float('num_epochs_per_decay', 0,
                      """Steps after which learning rate decays.""")
tf.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                      """Learning rate decay factor.""")
tf.flags.DEFINE_float('momentum', 0.9, """Momentum for training.""")
tf.flags.DEFINE_float('rmsprop_decay', 0.9, """Decay term for RMSProp.""")
tf.flags.DEFINE_float('rmsprop_momentum', 0.9, """Momentum in RMSProp.""")
tf.flags.DEFINE_float('rmsprop_epsilon', 1.0, """Epsilon term for RMSProp.""")
tf.flags.DEFINE_float('gradient_clip', None, """Gradient clipping magnitude.
                       Disabled by default.""")
tf.flags.DEFINE_float('weight_decay', 0.00004,
                      """Weight decay factor for training.""")

# Performance tuning flags.
tf.flags.DEFINE_boolean('winograd_nonfused', True,
                        """Enable/disable using the Winograd non-fused
                        algorithms.""")
tf.flags.DEFINE_boolean('sync_on_finish', False,
                        """Enable/disable whether the devices are synced after
                        each step.""")
tf.flags.DEFINE_boolean('staged_vars', False,
                        """whether the variables are staged from the main
                        computation""")
tf.flags.DEFINE_boolean('force_gpu_compatible', True,
                        """whether to enable force_gpu_compatible in
                        GPU_Options""")
# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable.  In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   replicated: each GPU has its own copy of the variables. To apply gradients,
#       nccl all-reduce or regular cross-device aggregation is used to replicate
#       the combined gradients to all towers (depending on --use_nccl option).
#   independent: each GPU has its own copy of the variables, and gradients are
#       not shared between towers. This can be used to check performance when no
#       data is moved between GPUs.
#   distributed_replicated: Distributed training only. Each GPU has a copy of
#       the variables, and updates its copy after the parameter servers are all
#       updated with the gradients from all servers. Only works with
#       cross_replica_sync=true. Unlike 'replicated', currently never uses
#       nccl all-reduce for replicating within a server.
tf.flags.DEFINE_string(
    'variable_update', 'parameter_server',
    ('The method for managing variables: '
     'parameter_server, replicated, distributed_replicated, independent'))
tf.flags.DEFINE_boolean(
    'use_nccl', True,
    'Whether to use nccl all-reduce primitives where possible')

# Distributed training flags.
tf.flags.DEFINE_string('job_name', '',
                       'One of "ps", "worker", "".  Empty for local training')
tf.flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
tf.flags.DEFINE_string('worker_hosts', '',
                       'Comma-separated list of target hosts')
tf.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
tf.flags.DEFINE_boolean('cross_replica_sync', True, '')

# Summary and Save & load checkpoints.
tf.flags.DEFINE_integer('summary_verbosity', 0,
                        """Verbosity level for summary ops. Pass 0 to disable
                        both summaries and checkpoints.""")
tf.flags.DEFINE_integer('save_summaries_steps', 0,
                        """How often to save summaries for trained models.
                        Pass 0 to disable summaries.""")
tf.flags.DEFINE_integer('save_model_secs', 0,
                        """How often to save trained models. Pass 0 to disable
                        checkpoints""")
tf.flags.DEFINE_string('train_dir', None,
                       """Path to session checkpoints.""")
tf.flags.DEFINE_string('eval_dir', '/tmp/tf_cnn_benchmarks/eval',
                       """Directory where to write eval event logs.""")
tf.flags.DEFINE_string('pretrain_dir', None,
                       """Path to pretrained session checkpoints.""")
tf.flags.DEFINE_string('result_storage', None,
                       """Specifies storage option for benchmark results.
                       None means results won't be stored.
                       'cbuild_benchmark_datastore' means results will be stored
                       in cbuild datastore (note: this option requires special
                       pemissions and meant to be used from cbuilds).""")
FLAGS = tf.flags.FLAGS

log_fn = print   # tf.logging.info


class GlobalStepWatcher(threading.Thread):
  """A helper class for globe_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op,
               start_at_global_step, end_at_global_step):
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
        log_fn('Starting real work at step %s at time %s' % (
            global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        log_fn('Finishing real work at step %s at time %s' % (
            global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def steps_per_second(self):
    return ((self.finish_step - self.start_step) /
            (self.finish_time - self.start_time))


class ConvNetBuilder(object):
  """Builder of cnn net."""

  def __init__(self,
               input_op,
               input_nchan,
               phase_train,
               data_format='NCHW',
               data_type=tf.float32):
    self.top_layer = input_op
    self.top_size = input_nchan
    self.phase_train = phase_train
    self.data_format = data_format
    self.data_type = data_type
    self.counts = defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = (
        'channels_last' if data_format == 'NHWC' else 'channels_first')

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           batch_norm=None,
           activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = conv_layers.conv2d(
            input_layer,
            num_out_channels, [k_height, k_width],
            strides=[d_height, d_width],
            padding=mode,
            data_format=self.channel_pos,
            use_bias=False)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = conv_layers.conv2d(
              input_layer,
              num_out_channels, [k_height, k_width],
              strides=[d_height, d_width],
              padding='SAME',
              data_format=self.channel_pos,
              use_bias=False)
        else:
          rate = 1  # Unused (for 'a trous' convolutions)
          kernel_size_effective = k_height + (k_width - 1) * (rate - 1)
          pad_total = kernel_size_effective - 1
          pad_beg = pad_total // 2
          pad_end = pad_total - pad_beg
          padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          input_layer = tf.pad(input_layer, padding)
          conv = conv_layers.conv2d(
              input_layer,
              num_out_channels, [k_height, k_width],
              strides=[d_height, d_width],
              padding='VALID',
              data_format=self.channel_pos,
              use_bias=False)
      if batch_norm is None:
        batch_norm = self.use_batch_norm
      if not batch_norm:
        biases = tf.get_variable(
            'biases', [num_out_channels], self.data_type,
            tf.constant_initializer(0.0))
        biased = tf.reshape(
            tf.nn.bias_add(
                conv, biases, data_format=self.data_format),
            conv.get_shape())
      else:
        self.top_layer = conv
        self.top_size = num_out_channels
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        conv1 = tf.nn.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = num_out_channels
      return conv1

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = 'mpool' + str(self.counts['mpool'])
    self.counts['mpool'] += 1
    pool = pooling_layers.max_pooling2d(
        input_layer, [k_height, k_width], [d_height, d_width],
        padding=mode,
        data_format=self.channel_pos,
        name=name)
    self.top_layer = pool
    return pool

  def apool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct an average pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = 'apool' + str(self.counts['apool'])
    self.counts['apool'] += 1
    pool = pooling_layers.average_pooling2d(
        input_layer, [k_height, k_width], [d_height, d_width],
        padding=mode,
        data_format=self.channel_pos,
        name=name)
    self.top_layer = pool
    return pool

  def reshape(self, shape, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    self.top_layer = tf.reshape(input_layer, shape)
    self.top_size = shape[-1]  # HACK This may not always work
    return self.top_layer

  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      kernel = tf.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.data_type,
          tf.random_normal_initializer(stddev=np.sqrt(init_factor /
                                                      (num_channels_in))))
      biases = tf.get_variable('biases', [num_out_channels],
                               self.data_type,
                               tf.constant_initializer(0.0))
      logits = tf.matmul(input_layer, kernel) + biases
      if activation == 'relu':
        affine1 = tf.nn.relu(logits, name=name)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

  def resnet_bottleneck_v1(self,
                           depth,
                           depth_bottleneck,
                           stride,
                           input_layer=None,
                           in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name = 'resnet_v1' + str(self.counts['resnet_v1'])
    self.counts['resnet_v1'] += 1
    with tf.variable_scope(name):
      if depth == in_size:
        if stride == 1:
          shortcut = input_layer
        else:
          shortcut = self.mpool(
              1,
              1,
              stride,
              stride,
              input_layer=input_layer,
              num_channels_in=in_size)
      else:
        shortcut = self.conv(
            depth,
            1,
            1,
            stride,
            stride,
            activation=None,
            input_layer=input_layer,
            num_channels_in=in_size)
      self.conv(
          depth_bottleneck,
          1,
          1,
          stride,
          stride,
          input_layer=input_layer,
          num_channels_in=in_size)
      self.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET')
      res = self.conv(depth, 1, 1, 1, 1, activation=None)
      output = tf.nn.relu(shortcut + res)
      self.top_layer = output
      self.top_size = depth
      return output

  def inception_module(self, name, cols, input_layer=None, in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name += str(self.counts[name])
    self.counts[name] += 1
    with tf.variable_scope(name):
      col_layers = []
      col_layer_sizes = []
      for c, col in enumerate(cols):
        col_layers.append([])
        col_layer_sizes.append([])
        for l, layer in enumerate(col):
          ltype, args = layer[0], layer[1:]
          kwargs = {
              'input_layer': input_layer,
              'num_channels_in': in_size
          } if l == 0 else {}
          if ltype == 'conv':
            self.conv(*args, **kwargs)
          elif ltype == 'mpool':
            self.mpool(*args, **kwargs)
          elif ltype == 'apool':
            self.apool(*args, **kwargs)
          elif ltype == 'share':  # Share matching layer from previous column
            self.top_layer = col_layers[c - 1][l]
            self.top_size = col_layer_sizes[c - 1][l]
          else:
            raise KeyError('Invalid layer type for inception module: \'%s\'' %
                           ltype)
          col_layers[c].append(self.top_layer)
          col_layer_sizes[c].append(self.top_size)
      catdim = 3 if self.data_format == 'NHWC' else 1
      self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
      self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
      return self.top_layer

  def residual(self, nout, net, scale=1.0):
    inlayer = self.top_layer
    net(self)
    self.conv(nout, 1, 1, activation=None)
    self.top_layer = tf.nn.relu(inlayer + scale * self.top_layer)

  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keep_dims=keep_dims, name=name)
    return self.top_layer

  def dropout(self, keep_prob=0.5, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'dropout' + str(self.counts['dropout'])
    with tf.variable_scope(name):
      if not self.phase_train:
        keep_prob = 1.0
      dropout = core_layers.dropout(input_layer, keep_prob)
      self.top_layer = dropout
      return dropout

  def batch_norm(self, input_layer=None, **kwargs):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    with tf.variable_scope(name) as scope:
      bn = tf.contrib.layers.batch_norm(
          input_layer, is_training=self.phase_train,
          fused=True, data_format=self.data_format,
          scope=scope, **kwargs)
    self.top_layer = bn
    return bn


def loss_function(logits, labels):
  # global cross_entropy # HACK TESTING
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def add_image_preprocessing(dataset, input_nchan, image_size, batch_size,
                            num_compute_devices, input_data_type,
                            resize_method, train):
  """Add image Preprocessing ops to tf graph."""
  if dataset is not None:
    preproc_train = preprocessing.ImagePreprocessor(
        image_size, image_size, batch_size,
        num_compute_devices, input_data_type, train=train,
        resize_method=resize_method)
    if train:
      subset = 'train'
    else:
      subset = 'validation'
    images, labels = preproc_train.minibatch(dataset, subset=subset)
    images_splits = images
    labels_splits = labels
    # Note: We force all datasets to 1000 to ensure even comparison
    #         This works because we use sparse_softmax_cross_entropy
    nclass = 1001
  else:
    nclass = 1001
    input_shape = [batch_size, image_size, image_size, input_nchan]
    images = tf.truncated_normal(
        input_shape,
        dtype=input_data_type,
        stddev=1e-1,
        name='synthetic_images')
    labels = tf.random_uniform(
        [batch_size],
        minval=1,
        maxval=nclass,
        dtype=tf.int32,
        name='synthetic_labels')
    # Note: This results in a H2D copy, but no computation
    # Note: This avoids recomputation of the random values, but still
    #         results in a H2D copy.
    images = tf.contrib.framework.local_variable(images, name='images')
    labels = tf.contrib.framework.local_variable(labels, name='labels')
    # Change to 0-based (don't use background class like Inception does)
    labels -= 1
    if num_compute_devices == 1:
      images_splits = [images]
      labels_splits = [labels]
    else:
      images_splits = tf.split(images, num_compute_devices, 0)
      labels_splits = tf.split(labels, num_compute_devices, 0)
  return nclass, images_splits, labels_splits


def create_config_proto():
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  return config


def get_mode_from_flags():
  """Determine which mode this script is running."""
  if FLAGS.forward_only and FLAGS.eval:
    raise ValueError('Only one of forward_only and eval flags is true')

  if FLAGS.eval:
    return 'evaluation'
  if FLAGS.forward_only:
    return 'forward-only'
  return 'training'


def benchmark_one_step(sess, fetches, step, batch_size,
                       step_train_times, trace_filename, summary_op=None):
  """Advance one step of benchmarking."""
  if trace_filename is not None and step == -1:
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

  if not FLAGS.forward_only:
    lossval = results[1]
  else:
    lossval = 0.

  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if step >= 0 and (step == 0 or (step + 1) % FLAGS.display_every == 0):
    log_fn('%i\t%s\t%.3f' % (
        step + 1, get_perf_timing_str(batch_size, step_train_times),
        lossval))
  if trace_filename is not None and step == -1:
    log_fn('Dumping trace to', trace_filename)
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    with open(trace_filename, 'w') as trace_file:
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
    return 'images/sec: %.1f +/- %.1f (jitter = %.1f)' % (
        speed_mean, speed_uncertainty, speed_jitter)
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
    raise RuntimeError('No checkpoint file found.')


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self):
    self.model = FLAGS.model
    self.model_conf = model_config.get_model_config(self.model)
    self.trace_filename = FLAGS.trace_file
    self.data_format = FLAGS.data_format
    self.num_batches = FLAGS.num_batches
    autotune_threshold = FLAGS.autotune_threshold if (
        FLAGS.autotune_threshold) else 1
    min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
    self.num_warmup_batches = FLAGS.num_warmup_batches if (
        FLAGS.num_warmup_batches) else max(10, min_autotune_warmup)
    self.graph_file = FLAGS.graph_file
    self.resize_method = FLAGS.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = FLAGS.num_gpus

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if FLAGS.batch_size > 0:
      self.model_conf.set_batch_size(FLAGS.batch_size)
    self.batch_size = self.model_conf.get_batch_size() * FLAGS.num_gpus

    # Use the learning rate from the command line if specified, otherwise use
    # the model's default learning rate, which must always be set.
    assert self.model_conf.get_learning_rate() > 0.0
    if FLAGS.learning_rate is not None:
      self.model_conf.set_learning_rate(FLAGS.learning_rate)

    self.job_name = FLAGS.job_name  # "" for local training
    self.ps_hosts = FLAGS.ps_hosts.split(',')
    self.worker_hosts = FLAGS.worker_hosts.split(',')
    self.dataset = None
    self.data_name = FLAGS.data_name
    if FLAGS.data_dir is not None:
      if self.data_name is None:
        if 'imagenet' in FLAGS.data_dir:
          self.data_name = 'imagenet'
        elif 'flowers' in FLAGS.data_dir:
          self.data_name = 'flowers'
        else:
          raise ValueError('Could not identify name of dataset. '
                           'Please specify with --data_name option.')
      if self.data_name == 'imagenet':
        self.dataset = datasets.ImagenetData(FLAGS.data_dir)
      elif self.data_name == 'flowers':
        self.dataset = datasets.FlowersData(FLAGS.data_dir)
      else:
        raise ValueError('Unknown dataset. Must be one of imagenet or flowers.')

    self.local_parameter_device_flag = FLAGS.local_parameter_device
    if self.job_name:
      self.task_index = FLAGS.task_index
      self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts,
                                           'worker': self.worker_hosts})
      self.server = None

      if not self.server:
        self.server = tf.train.Server(self.cluster, job_name=self.job_name,
                                      task_index=self.task_index,
                                      config=create_config_proto(),
                                      protocol=FLAGS.server_protocol)
      worker_prefix = '/job:worker/task:%s' % self.task_index
      self.param_server_device = tf.train.replica_device_setter(
          worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
      # This device on which the queues for managing synchronization between
      # servers should be stored.
      num_ps = len(self.ps_hosts)
      self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i
                                 for i in range(num_ps)]
    else:
      self.task_index = 0
      self.cluster = None
      self.server = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % FLAGS.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = ['%s/%s:%i' % (worker_prefix, FLAGS.device, i)
                        for i in xrange(FLAGS.num_gpus)]

    if FLAGS.staged_vars and FLAGS.variable_update != 'parameter_server':
      raise ValueError('staged_vars for now is only supported with '
                       '--variable_update=parameter_server')

    if FLAGS.variable_update == 'parameter_server':
      if self.job_name:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
        else:
          self.variable_mgr = (
              variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
      else:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
              self)
    elif FLAGS.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid --variable_update in distributed mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, FLAGS.use_nccl)
    elif FLAGS.variable_update == 'distributed_replicated':
      if not self.job_name:
        raise ValueError('Invalid --variable_update in local mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif FLAGS.variable_update == 'independent':
      if self.job_name:
        raise ValueError('Invalid --variable_update in distributed mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError('Invalid --variable_update: %s' % FLAGS.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      self.global_step_device = self.param_server_device
    else:
      self.global_step_device = self.cpu_device

  def print_info(self):
    """Print basic information."""
    log_fn('Model:       %s' % self.model)
    log_fn('Mode:        %s' % get_mode_from_flags())
    log_fn('Batch size:  %s global' % self.batch_size)
    log_fn('             %s per device' % (self.batch_size / len(self.devices)))
    log_fn('Devices:     %s' % self.raw_devices)
    log_fn('Data format: %s' % self.data_format)
    log_fn('Optimizer:   %s' % FLAGS.optimizer)
    log_fn('Variables:   %s' % FLAGS.variable_update)
    if FLAGS.variable_update == 'replicated':
      log_fn('Use NCCL:    %s' % FLAGS.use_nccl)
    if self.job_name:
      log_fn('Sync:        %s' % FLAGS.cross_replica_sync)
    if FLAGS.staged_vars:
      log_fn('Staged vars: %s' % FLAGS.staged_vars)
    log_fn('==========')

  def run(self):
    if FLAGS.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.server.join()
      return

    with tf.Graph().as_default():
      if FLAGS.eval:
        self._eval_cnn()
      else:
        self._benchmark_cnn()

  def _eval_cnn(self):
    """Evaluate the model from a checkpoint using validation dataset."""
    (enqueue_ops, fetches) = self._build_model()
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           tf.get_default_graph())
    target = ''
    with tf.Session(target=target, config=create_config_proto()) as sess:
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i+1)])
      if FLAGS.train_dir is None:
        raise ValueError('Trained model directory not specified')
      global_step = load_checkpoint(saver, sess, FLAGS.train_dir)

      start_time = time.time()
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_eval_count = self.num_batches * self.batch_size
      for step in xrange(self.num_batches):
        results = sess.run(fetches)
        count_top_1 += results[0]
        count_top_5 += results[1]
        if (step + 1) % FLAGS.display_every == 0:
          duration = time.time() - start_time
          examples_per_sec = self.batch_size * self.num_batches / duration
          log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
          start_time = time.time()
      precision_at_1 = count_top_1 / total_eval_count
      recall_at_5 = count_top_5 / total_eval_count
      summary = tf.Summary()
      summary.value.add(tag='eval/Accuracy@1', simple_value=precision_at_1)
      summary.value.add(tag='eval/Recall@5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)
      log_fn('Precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
             (precision_at_1, recall_at_5, total_eval_count))

  def _benchmark_cnn(self):
    """Run cnn in benchmark mode. When forward_only on, it forwards CNN."""
    (enqueue_ops, fetches) = self._build_model()
    main_fetch_group = tf.group(*fetches)
    execution_barrier = None
    if self.job_name and not FLAGS.cross_replica_sync:
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.contrib.framework.get_global_step()
    with tf.device(self.global_step_device):
      with tf.control_dependencies([main_fetch_group]):
        inc_global_step = global_step.assign_add(1)
        fetches.append(inc_global_step)

    if self.job_name and FLAGS.cross_replica_sync:
      # Block all replicas until all replicas are ready for next step.
      fetches.append(self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group]))

    variable_mgr_post_init_ops = self.variable_mgr.get_post_init_ops()
    if variable_mgr_post_init_ops:
      post_init_op_group = tf.group(*variable_mgr_post_init_ops)
    else:
      post_init_op_group = None

    local_var_init_op = tf.local_variables_initializer()
    summary_op = tf.summary.merge_all()
    is_chief = (not self.job_name or self.task_index == 0)
    summary_writer = None
    if (is_chief and FLAGS.summary_verbosity and
        FLAGS.train_dir and
        FLAGS.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             tf.get_default_graph())

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=FLAGS.train_dir,
        saver=tf.train.Saver(tf.global_variables()),
        global_step=global_step,
        summary_op=None,
        save_model_secs=FLAGS.save_model_secs,
        summary_writer=summary_writer)

    step_train_times = []
    with sv.managed_session(
        master=self.server.target if self.server else '',
        config=create_config_proto(),
        start_standard_services=FLAGS.summary_verbosity > 0) as sess:
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i+1)])
      sess.run(local_var_init_op)
      if post_init_op_group:
        sess.run(post_init_op_group)

      init_global_step = 0
      if FLAGS.pretrain_dir is not None:
        init_global_step = load_checkpoint(sv.saver, sess, FLAGS.pretrain_dir)
      global_step_watcher = GlobalStepWatcher(
          sess, global_step,
          len(self.worker_hosts) * self.num_warmup_batches + init_global_step,
          len(self.worker_hosts) * (
              self.num_warmup_batches + self.num_batches) - 1)
      global_step_watcher.start()

      if self.graph_file is not None:
        path, filename = os.path.split(self.graph_file)
        as_text = filename.endswith('txt')
        log_fn('Writing GraphDef as %s to %s' % (
            'text' if as_text else 'binary', self.graph_file))
        tf.train.write_graph(sess.graph_def, path, filename, as_text)

      log_fn('Running warm up')
      local_step = -1 * self.num_warmup_batches

      if FLAGS.cross_replica_sync and FLAGS.job_name:
        # In cross-replica sync mode, all workers must run the same number of
        # local steps, or else the workers running the extra step will block.
        done_fn = lambda: local_step == self.num_batches
      else:
        done_fn = lambda: global_step_watcher.done()
      while not done_fn():
        if local_step == 0:
          log_fn('Done warm up')
          if execution_barrier:
            log_fn('Waiting for other replicas to finish warm up')
            assert global_step_watcher.start_time == 0
            sess.run([execution_barrier])

          log_fn('Step\tImg/sec\tloss')
          assert len(step_train_times) == self.num_warmup_batches
          step_train_times = []  # reset to ignore warm up batches
        if (summary_writer and
            (local_step + 1) % FLAGS.save_summaries_steps == 0):
          fetch_summary = summary_op
        else:
          fetch_summary = None
        summary_str = benchmark_one_step(
            sess, fetches, local_step, self.batch_size, step_train_times,
            self.trace_filename, fetch_summary)
        if summary_str is not None and is_chief:
          sv.summary_computed(sess, summary_str)
        local_step += 1
      # Waits for the global step to be done, regardless of done_fn.
      while not global_step_watcher.done():
        time.sleep(.25)
      images_per_sec = global_step_watcher.steps_per_second() * self.batch_size
      log_fn('-' * 64)
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)
      if is_chief:
        store_benchmarks({'total_images_per_sec': images_per_sec})
      # Save the model checkpoint.
      if FLAGS.train_dir is not None and is_chief:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        if not gfile.Exists(FLAGS.train_dir):
          gfile.MakeDirs(FLAGS.train_dir)
        sv.saver.save(sess, checkpoint_path, global_step)

      if execution_barrier:
        # Wait for other workers to reach the end, so this worker doesn't
        # go away underneath them.
        sess.run([execution_barrier])
    sv.stop()

  def _build_model(self):
    """Build the TensorFlow graph."""
    image_size = self.model_conf.get_image_size()
    data_type = tf.float32
    input_data_type = tf.float32
    input_nchan = 3
    tf.set_random_seed(1234)
    np.random.seed(4321)
    phase_train = not (FLAGS.eval or FLAGS.forward_only)

    log_fn('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    enqueue_ops = []
    gpu_copy_stage_ops = []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    use_synthetic_gpu_images = (self.dataset is None)

    with tf.device(self.global_step_device):
      global_step = tf.contrib.framework.get_or_create_global_step()

    # Build the processing and model for the worker.
    with tf.device(self.cpu_device):
      nclass, images_splits, labels_splits = add_image_preprocessing(
          self.dataset, input_nchan, image_size, self.batch_size,
          len(self.devices), input_data_type, self.resize_method,
          not FLAGS.eval)

    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with self.variable_mgr.create_outer_variable_scope(
          device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
        results = self.add_forward_pass_and_gradients(
            images_splits[device_num], labels_splits[device_num], nclass,
            phase_train, device_num, input_data_type, data_type, input_nchan,
            use_synthetic_gpu_images, gpu_copy_stage_ops, gpu_compute_stage_ops,
            gpu_grad_stage_ops)
        if phase_train:
          losses.append(results[0])
          device_grads.append(results[1])
        else:
          all_logits.append(results[0])
          all_top_1_ops.append(results[1])
          all_top_5_ops.append(results[2])

        if self.variable_mgr.retain_tower_updates(device_num):
          # Retain the Batch Normalization updates operations only from the
          # first tower. Ideally, we should grab the updates from all towers but
          # these stats accumulate extremely fast so we can ignore the other
          # stats from the other towers without significant detriment.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

    if not update_ops:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
    enqueue_ops.append(tf.group(*gpu_copy_stage_ops))
    if self.variable_mgr.supports_staged_vars():
      for staging_ops in self.variable_mgr.staging_vars_on_devices:
        gpu_compute_stage_ops.extend(
            [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
    enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
    if gpu_grad_stage_ops:
      staging_delta_ops += gpu_grad_stage_ops
    if staging_delta_ops:
      enqueue_ops.append(tf.group(*(staging_delta_ops)))

    if not phase_train:
      if FLAGS.forward_only:
        all_logits = tf.concat(all_logits, 0)
        fetches = [all_logits] + enqueue_ops
      else:
        all_top_1_ops = tf.reduce_sum(all_top_1_ops)
        all_top_5_ops = tf.reduce_sum(all_top_5_ops)
        fetches = [all_top_1_ops, all_top_5_ops] + enqueue_ops
      return (enqueue_ops, fetches)
    extra_nccl_ops = []
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        total_loss = tf.reduce_mean(losses)
        avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

        gradient_clip = FLAGS.gradient_clip
        learning_rate = self.model_conf.get_learning_rate()
        if self.dataset and FLAGS.num_epochs_per_decay > 0:
          num_batches_per_epoch = (
              self.dataset.num_examples_per_epoch() / self.batch_size)
          decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

          # Decay the learning rate exponentially based on the number of steps.
          learning_rate = tf.train.exponential_decay(
              FLAGS.learning_rate, global_step,
              decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)

        if gradient_clip is not None:
          clipped_grads = [
              (tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var)
              for grad, var in avg_grads
          ]
        else:
          clipped_grads = avg_grads

        if FLAGS.optimizer == 'momentum':
          opt = tf.train.MomentumOptimizer(
              learning_rate, FLAGS.momentum, use_nesterov=True)
        elif FLAGS.optimizer == 'sgd':
          opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif FLAGS.optimizer == 'rmsprop':
          opt = tf.train.RMSPropOptimizer(learning_rate, FLAGS.rmsprop_decay,
                                          momentum=FLAGS.rmsprop_momentum,
                                          epsilon=FLAGS.rmsprop_epsilon)
        else:
          raise ValueError('Optimizer "%s" was not recognized', FLAGS.optimizer)

        self.variable_mgr.append_apply_gradients_ops(
            gradient_state, opt, clipped_grads, training_ops)
    train_op = tf.group(*(training_ops + update_ops + extra_nccl_ops))

    with tf.device(self.cpu_device):
      if self.task_index == 0 and FLAGS.summary_verbosity > 0:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        for grad, var in avg_grads:
          if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
        for var in tf.trainable_variables():
          tf.summary.histogram(var.op.name, var)
    fetches = [train_op, total_loss] + enqueue_ops
    return (enqueue_ops, fetches)

  def add_forward_pass_and_gradients(
      self, host_images, host_labels, nclass, phase_train, device_num,
      input_data_type, data_type, input_nchan, use_synthetic_gpu_images,
      gpu_copy_stage_ops, gpu_compute_stage_ops, gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    if not use_synthetic_gpu_images:
      with tf.device(self.cpu_device):
        images_shape = host_images.get_shape()
        labels_shape = host_labels.get_shape()
        gpu_copy_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.int32],
            shapes=[images_shape, labels_shape])
        gpu_copy_stage_op = gpu_copy_stage.put(
            [host_images, host_labels])
        gpu_copy_stage_ops.append(gpu_copy_stage_op)
        host_images, host_labels = gpu_copy_stage.get()

    with tf.device(self.raw_devices[device_num]):
      if not use_synthetic_gpu_images:
        gpu_compute_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.int32],
            shapes=[images_shape, labels_shape]
        )
        # The CPU-to-GPU copy is triggered here.
        gpu_compute_stage_op = gpu_compute_stage.put(
            [host_images, host_labels])
        images, labels = gpu_compute_stage.get()
        images = tf.reshape(images, shape=images_shape)
        gpu_compute_stage_ops.append(gpu_compute_stage_op)
      else:
        # Minor hack to avoid H2D copy when using synthetic data
        images = tf.truncated_normal(
            host_images.get_shape(),
            dtype=input_data_type,
            stddev=1e-1,
            name='synthetic_images')
        images = tf.contrib.framework.local_variable(
            images, name='gpu_cached_images')
        labels = host_labels

    with tf.device(self.devices[device_num]):
      # Rescale to [0, 1)
      images *= 1. / 256
      # Rescale to [-1,1] instead of [0, 1)
      images = tf.subtract(images, 0.5)
      images = tf.multiply(images, 2.0)

      if self.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
      if input_data_type != data_type:
        images = tf.cast(images, data_type)
      network = ConvNetBuilder(
          images, input_nchan, phase_train, self.data_format, data_type)
      self.model_conf.add_inference(network)
      # Add the final fully-connected class layer
      logits = network.affine(nclass, activation='linear')
      if not phase_train:
        top_1_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
        top_5_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
        return (logits, top_1_op, top_5_op)
      loss = loss_function(logits, labels)
      params = self.variable_mgr.trainable_variables_on_device(device_num)
      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
      weight_decay = FLAGS.weight_decay
      if weight_decay is not None and weight_decay != 0.:
        loss += weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      grads = tf.gradients(loss, params, aggregation_method=aggmeth)

      if FLAGS.staged_vars:
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
          device_num, writable=True)
      gradvars = list(zip(grads, param_refs))
      return (loss, gradvars)

  def add_sync_queues_and_barrier(self, name_prefix,
                                  enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      an op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    num_workers = self.cluster.num_tasks('worker')
    with tf.device(self.sync_queue_devices[
        self.sync_queue_counter % len(self.sync_queue_devices)]):
      sync_queues = [
          tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(num_workers)]
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


def store_benchmarks(names_to_values):
  if FLAGS.result_storage:
    benchmark_storage.store_benchmark(names_to_values, FLAGS.result_storage)


def main(_):
  if FLAGS.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  if FLAGS.autotune_threshold:
    os.environ['TF_AUTOTUNE_THRESHOLD'] = str(FLAGS.autotune_threshold)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(FLAGS.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)


  bench = BenchmarkCNN()

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()


if __name__ == '__main__':
  tf.app.run()
