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
"""DeepSpeech2 model configuration.

References:
  https://arxiv.org/abs/1512.02595
  Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import constants
from cnn_util import log_fn
from models import model as model_lib


class DeepSpeechDecoder(object):
  """Greedy decoder implementation for Deep Speech model."""

  def __init__(self, labels, blank_index=28):
    """Decoder initialization.

    Arguments:
      labels: a string specifying the speech labels for the decoder to use.
      blank_index: an integer specifying index for the blank character. Defaults
        to 28.
    """
    self.labels = labels
    self.blank_index = blank_index
    self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

  def convert_to_string(self, sequence):
    """Convert a sequence of indexes into corresponding string."""
    return ''.join([self.int_to_char[i] for i in sequence])

  def wer(self, decode, target):
    """Computes the Word Error Rate (WER).

    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.

    Args:
      decode: string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number for the WER of the current decode-target pair.
    """
    try:
      from nltk.metrics import distance  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      if 'nltk.metrics' not in e.message:
        raise
      raise ImportError('To use the experimental deepspeech model, you must '
                        'pip install -U nltk')

    # Map each word to a new char.
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))

  def cer(self, decode, target):
    """Computes the Character Error Rate (CER).

    CER is defined as the edit distance between the two given strings.

    Args:
      decode: a string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number denoting the CER for the current sentence pair.
    """
    try:
      from nltk.metrics import distance  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      if 'nltk.metrics' not in e.message:
        raise
      raise ImportError('To use the experimental deepspeech model, you must '
                        'pip install -U nltk')
    return distance.edit_distance(decode, target)

  def decode(self, char_indexes):
    """Decode the best guess from logits using greedy algorithm."""
    # Merge repeated chars.
    merge = [k for k, _ in itertools.groupby(char_indexes)]
    # Remove the blank index in the decoded sequence.
    merge_remove_blank = []
    for k in merge:
      if k != self.blank_index:
        merge_remove_blank.append(k)

    return self.convert_to_string(merge_remove_blank)

  def decode_logits(self, logits):
    """Decode the best guess from logits using greedy algorithm."""
    # Choose the class with maximimum probability.
    best = list(np.argmax(logits, axis=1))
    return self.decode(best)


class DeepSpeech2Model(model_lib.Model):
  """Define DeepSpeech2 model."""

  # Supported rnn cells.
  SUPPORTED_RNNS = {
      'lstm': tf.nn.rnn_cell.BasicLSTMCell,
      'rnn': tf.nn.rnn_cell.RNNCell,
      'gru': tf.nn.rnn_cell.GRUCell,
  }

  # Parameters for batch normalization.
  BATCH_NORM_EPSILON = 1e-5
  BATCH_NORM_DECAY = 0.997

  # Filters of convolution layer
  CONV_FILTERS = 32

  def __init__(self,
               num_rnn_layers=5,
               rnn_type='lstm',
               is_bidirectional=True,
               rnn_hidden_size=800,
               use_bias=True,
               params=None):
    """Initialize DeepSpeech2 model.

    Args:
      num_rnn_layers: an integer, the number of rnn layers (default: 5).
      rnn_type: a string, one of the supported rnn cells: gru, rnn or lstm.
      is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
      rnn_hidden_size: an integer for the number of hidden units in the RNN
        cell.
      use_bias: a boolean specifying whether to use a bias in the last fc layer.
      params: the params from BenchmarkCNN.
    """
    super(DeepSpeech2Model, self).__init__(
        'deepspeech2',
        batch_size=128,
        learning_rate=0.0005,
        fp16_loss_scale=128,
        params=params)
    self.num_rnn_layers = num_rnn_layers
    self.rnn_type = rnn_type
    self.is_bidirectional = is_bidirectional
    self.rnn_hidden_size = rnn_hidden_size
    self.use_bias = use_bias
    self.num_feature_bins = 161
    self.max_time_steps = 3494
    self.max_label_length = 576

  def _batch_norm(self, inputs, training):
    """Batch normalization layer.

    Note that the momentum to use will affect validation accuracy over time.
    Batch norm has different behaviors during training/evaluation. With a large
    momentum, the model takes longer to get a near-accurate estimation of the
    moving mean/variance over the entire training dataset, which means we need
    more iterations to see good evaluation results. If the training data is
    evenly distributed over the feature space, we can also try setting a smaller
    momentum (such as 0.1) to get good evaluation result sooner.

    Args:
      inputs: input data for batch norm layer.
      training: a boolean to indicate if it is in training stage.

    Returns:
      tensor output from batch norm layer.
    """
    return tf.layers.batch_normalization(
        inputs=inputs,
        momentum=DeepSpeech2Model.BATCH_NORM_DECAY,
        epsilon=DeepSpeech2Model.BATCH_NORM_EPSILON,
        fused=True,
        training=training)

  def _conv_bn_layer(self, inputs, padding, filters, kernel_size, strides,
                     layer_id, training):
    """Defines 2D convolutional + batch normalization layer.

    Args:
      inputs: input data for convolution layer.
      padding: padding to be applied before convolution layer.
      filters: an integer, number of output filters in the convolution.
      kernel_size: a tuple specifying the height and width of the 2D convolution
        window.
      strides: a tuple specifying the stride length of the convolution.
      layer_id: an integer specifying the layer index.
      training: a boolean to indicate which stage we are in (training/eval).

    Returns:
      tensor output from the current layer.
    """
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence is shorter
    # than the label length.
    inputs = tf.pad(
        inputs,
        [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid',
        use_bias=False,
        activation=tf.nn.relu6,
        name='cnn_{}'.format(layer_id))
    return self._batch_norm(inputs, training)

  def _rnn_layer(self, inputs, rnn_cell, rnn_hidden_size, layer_id,
                 use_batch_norm, is_bidirectional, training):
    """Defines a batch normalization + rnn layer.

    Args:
      inputs: input tensors for the current layer.
      rnn_cell: RNN cell instance to use.
      rnn_hidden_size: an integer for the dimensionality of the rnn output
        space.
      layer_id: an integer for the index of current layer.
      use_batch_norm: a boolean specifying whether to perform batch
        normalization on input states.
      is_bidirectional: a boolean specifying whether the rnn layer is
        bi-directional.
      training: a boolean to indicate which stage we are in (training/eval).

    Returns:
      tensor output for the current layer.
    """
    if use_batch_norm:
      inputs = self._batch_norm(inputs, training)

    # Construct forward/backward RNN cells.
    fw_cell = rnn_cell(
        num_units=rnn_hidden_size, name='rnn_fw_{}'.format(layer_id))

    if is_bidirectional:
      bw_cell = rnn_cell(
          num_units=rnn_hidden_size, name='rnn_bw_{}'.format(layer_id))
      outputs, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=fw_cell,
          cell_bw=bw_cell,
          inputs=inputs,
          dtype=tf.float32,
          swap_memory=True)
      rnn_outputs = tf.concat(outputs, -1)
    else:
      rnn_outputs = tf.nn.dynamic_rnn(
          fw_cell, inputs, dtype=tf.float32, swap_memory=True)

    return rnn_outputs

  def get_input_data_types(self, subset):
    """Returns the list of data types of the inputs."""
    del subset  # Same data types for both train and validation subsets.
    return [self.data_type, tf.int32, tf.int32, tf.int32]

  def get_input_shapes(self, subset):
    """Returns the list of shapes of the padded inputs."""
    del subset  # Same shapes for both train and validation subsets
    return [
        [self.batch_size, self.max_time_steps, self.num_feature_bins, 1],
        [self.batch_size, self.max_label_length],
        [self.batch_size, 1],
        [self.batch_size, 1],
    ]

  def get_synthetic_inputs(self, input_name, nclass):
    inputs = tf.random_uniform(self.get_input_shapes('train')[0],
                               dtype=self.get_input_data_types('train')[0])
    inputs = tf.contrib.framework.local_variable(inputs, name=input_name)
    labels = tf.convert_to_tensor(
        np.random.randint(28, size=[self.batch_size, self.max_label_length]))
    input_lengths = tf.convert_to_tensor(
        [self.max_time_steps] * self.batch_size)
    label_lengths = tf.convert_to_tensor(
        [self.max_label_length] * self.batch_size)
    return [inputs, labels, input_lengths, label_lengths]

  # TODO(laigd): support fp16.
  # TODO(laigd): support multiple gpus.
  def build_network(self, inputs, phase_train=True, nclass=29):
    """Builds the forward pass of the deepspeech2 model.

    Args:
      inputs: The input list of the model.
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the input spectrogram can belong to.

    Returns:
      A BuildNetworkResult which contains the logits and model-specific extra
        information.
    """
    inputs = inputs[0]  # Get the spectrogram feature.

    # Two cnn layers.
    inputs = self._conv_bn_layer(
        inputs,
        padding=(20, 5),
        filters=DeepSpeech2Model.CONV_FILTERS,
        kernel_size=(41, 11),
        strides=(2, 2),
        layer_id=1,
        training=phase_train)

    inputs = self._conv_bn_layer(
        inputs,
        padding=(10, 5),
        filters=DeepSpeech2Model.CONV_FILTERS,
        kernel_size=(21, 11),
        strides=(2, 1),
        layer_id=2,
        training=phase_train)

    # output of conv_layer2 with the shape of
    # [batch_size (N), times (T), features (F), channels (C)].
    # Convert the conv output to rnn input.

    # batch_size = tf.shape(inputs)[0]
    feat_size = inputs.get_shape().as_list()[2]
    inputs = tf.reshape(
        inputs,
        [self.batch_size, -1, feat_size * DeepSpeech2Model.CONV_FILTERS])

    # RNN layers.
    rnn_cell = DeepSpeech2Model.SUPPORTED_RNNS[self.rnn_type]
    for layer_counter in xrange(self.num_rnn_layers):
      # No batch normalization on the first layer.
      use_batch_norm = (layer_counter != 0)
      inputs = self._rnn_layer(inputs, rnn_cell, self.rnn_hidden_size,
                               layer_counter + 1, use_batch_norm,
                               self.is_bidirectional, phase_train)

    # FC layer with batch norm.
    inputs = self._batch_norm(inputs, phase_train)
    logits = tf.layers.dense(inputs, nclass, use_bias=self.use_bias)

    return model_lib.BuildNetworkResult(logits=logits, extra_info=None)

  def loss_function(self, inputs, build_network_result):
    """Computes the ctc loss for the current batch of predictions.

    Args:
      inputs: the input list of the model.
      build_network_result: a BuildNetworkResult returned by build_network().

    Returns:
      The loss tensor of the model.
    """
    logits = build_network_result.logits
    actual_time_steps = inputs[2]
    probs = tf.nn.softmax(logits)
    ctc_time_steps = tf.shape(probs)[1]
    ctc_input_length = tf.to_float(
        tf.multiply(actual_time_steps, ctc_time_steps))
    ctc_input_length = tf.to_int32(
        tf.floordiv(ctc_input_length, tf.to_float(self.max_time_steps)))

    label_length = inputs[3]
    label_length = tf.to_int32(tf.squeeze(label_length))
    ctc_input_length = tf.to_int32(tf.squeeze(ctc_input_length))

    labels = inputs[1]
    sparse_labels = tf.to_int32(
        tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length))
    y_pred = tf.log(
        tf.transpose(probs, perm=[1, 0, 2]) + tf.keras.backend.epsilon())

    losses = tf.expand_dims(
        tf.nn.ctc_loss(
            labels=sparse_labels,
            inputs=y_pred,
            sequence_length=ctc_input_length,
            ignore_longer_outputs_than_inputs=True),
        axis=1)
    loss = tf.reduce_mean(losses)
    return loss

  PROBABILITY_TENSOR = 'deepspeech2_prob'
  LABEL_TENSOR = 'deepspeech2_label'

  def accuracy_function(self, inputs, logits):
    """Returns the ops to evaluate the model performance."""
    # Get probabilities of each predicted class
    probs = tf.nn.softmax(logits)
    assert probs.shape.as_list()[0] == self.batch_size
    return {
        (constants.UNREDUCED_ACCURACY_OP_PREFIX + self.PROBABILITY_TENSOR):
            probs,
        (constants.UNREDUCED_ACCURACY_OP_PREFIX + self.LABEL_TENSOR):
            inputs[1],
    }

  def postprocess(self, results):
    """Postprocess results returned from model in Python."""
    probs = results[self.PROBABILITY_TENSOR]

    total_wer, total_cer = 0, 0
    speech_labels = " abcdefghijklmnopqrstuvwxyz'-"
    greedy_decoder = DeepSpeechDecoder(speech_labels)

    # Evaluate the performance using WER (Word Error Rate) and CER (Character
    # Error Rate) as metrics.
    targets = results[self.LABEL_TENSOR]  # The ground truth transcript
    for i in range(self.batch_size):
      # Decode string.
      predicted_str = greedy_decoder.decode_logits(probs[i])
      expected_str = greedy_decoder.decode(targets[i])
      # Compute CER.
      total_cer += (greedy_decoder.cer(predicted_str, expected_str) /
                    len(expected_str))
      # Compute WER.
      total_wer += (greedy_decoder.wer(predicted_str, expected_str) /
                    len(expected_str.split()))

    # Get mean value
    total_cer /= self.batch_size
    total_wer /= self.batch_size

    log_fn('total CER: {:f}; total WER: {:f}; total example: {:d}.'.format(
        total_cer, total_wer, self.batch_size))
    # TODO(laigd): get rid of top_N_accuracy bindings in benchmark_cnn.py
    return {'top_1_accuracy': 0., 'top_5_accuracy': 0.}
