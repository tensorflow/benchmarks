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
"""CNN builder."""

from __future__ import print_function

from collections import defaultdict
import contextlib

import numpy as np

import tensorflow as tf

from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers


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
    self.channel_pos = ('channels_last'
                        if data_format == 'NHWC' else 'channels_first')
    self.aux_top_layer = None
    self.aux_top_size = 0

  @contextlib.contextmanager
  def switch_to_aux_top_layer(self):
    """Context that construct cnn in the auxiliary arm."""
    if self.aux_top_layer is None:
      raise RuntimeError('Empty auxiliary top layer in the network.')
    saved_top_layer = self.top_layer
    saved_top_size = self.top_size
    self.top_layer = self.aux_top_layer
    self.top_size = self.aux_top_size
    yield
    self.aux_top_layer = self.top_layer
    self.aux_top_size = self.top_size
    self.top_layer = saved_top_layer
    self.top_size = saved_top_size

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           use_batch_norm=None,
           stddev=None,
           activation='relu',
           bias=0.0):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    kernel_initializer = None
    if stddev is not None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
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
            kernel_initializer=kernel_initializer,
            use_bias=False)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = conv_layers.conv2d(
              input_layer,
              num_out_channels, [k_height, k_width],
              strides=[d_height, d_width],
              padding='SAME',
              data_format=self.channel_pos,
              kernel_initializer=kernel_initializer,
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
              kernel_initializer=kernel_initializer,
              use_bias=False)
      if use_batch_norm is None:
        use_batch_norm = self.use_batch_norm
      if not use_batch_norm:
        if bias is not None:
          biases = tf.get_variable('biases', [num_out_channels], self.data_type,
                                   tf.constant_initializer(bias))
          biased = tf.reshape(
              tf.nn.bias_add(conv, biases, data_format=self.data_format),
              conv.get_shape())
        else:
          biased = conv
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
             bias=0.0,
             stddev=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      stddev = stddev or np.sqrt(init_factor / num_channels_in)
      kernel = tf.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.data_type,
          tf.truncated_normal_initializer(stddev=stddev))
      biases = tf.get_variable('biases', [num_out_channels], self.data_type,
                               tf.constant_initializer(bias))
      logits = tf.nn.xw_plus_b(input_layer, kernel, biases)
      if activation == 'relu':
        affine1 = tf.nn.relu(logits, name=name)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

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
            raise KeyError(
                'Invalid layer type for inception module: \'%s\'' % ltype)
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
          input_layer,
          is_training=self.phase_train,
          fused=True,
          data_format=self.data_format,
          scope=scope,
          **kwargs)
    self.top_layer = bn
    return bn

  def lrn(self, depth_radius, bias, alpha, beta):
    """Adds a local response normalization layer."""
    name = 'lrn' + str(self.counts['lrn'])
    self.counts['lrn'] += 1
    self.top_layer = tf.nn.lrn(
        self.top_layer, depth_radius, bias, alpha, beta, name=name)
    return self.top_layer
