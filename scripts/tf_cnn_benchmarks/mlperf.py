# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Contains functions related to MLPerf compliance.

MLPerf requires submissions to log what the benchmark does, in order to verify
that the benchmark meets the MLPerf requirements. This module contains a global
object `logger` that is used by other files to log what tf_cnn_benchmarks does
for compliance.

By default, `logger` does nothing, as the MLPerf compliance logs are verbose and
unnecessary if one is not concerned about MLPerf compliance. The logger can be
enabled by using the `mlperf_logger` context manager.

To enable the logger with `mlperf_logger`, the MLPerf compliance library at
https://github.com/mlperf/training/tree/master/compliance is required. If
the logger is not enabled, the library is not needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import contextlib
import os

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  # Not all users have the MLPerf compliance library, so we don't want to
  # unconditionally crash if these imports fail.
  from mlperf_compliance import mlperf_log
  from mlperf_compliance import resnet_log_helper
  from mlperf_compliance import tags
  import_successful = True
except ImportError:
  # The logger cannot be enabled in this case since the MLPerf library isn't
  # found. We return empty strings from the `tags` attribute so that
  # the benchmark can still run without crashing. This empty tags are passed
  # to an instance of `NullMlPerfLogger`, which does not log anything and
  # ignores the tag values.

  class _Tags(object):

    def __getattr__(self, item):
      return ''
  tags = _Tags()
  import_successful = False
# pylint: enable=g-import-not-at-top


class MlPerfLogger(object):
  """Logs various aspects about a benchmark run for MLPerf compliance."""

  def __init__(self, model):
    mlperf_log.ROOT_DIR_RESNET = os.path.split(os.path.abspath(__file__))[0]
    self.model = model
    model_to_fn = {
        'resnet50': mlperf_log.resnet_print
    }
    try:
      self._log_fn = model_to_fn[model]
    except KeyError:
      raise ValueError('--ml_perf_compliance_logging is only compatible when '
                       '--model is one of the following: ' +
                       ', '.join(model_to_fn.keys()))

  def log(self, key, value=None, stack_offset=2):
    self._log_fn(key, value, stack_offset)

  def log_deferred_tensor_value(self, key, tensor_value, stack_offset=2,
                                every_n=1, first_n=None):
    del tensor_value, every_n, first_n
    self._log_fn(key, stack_offset=stack_offset, deferred=True)
    # TODO(reedwm): Log the deferred value.

  def log_max_pool(self, input_tensor, output_tensor):
    if self.model == 'resnet50':
      resnet_log_helper.log_max_pool(input_tensor, output_tensor)

  def log_begin_block(self, input_tensor, block_type):
    if self.model == 'resnet50':
      resnet_log_helper.log_begin_block(input_tensor, block_type)

  def log_end_block(self, output_tensor):
    if self.model == 'resnet50':
      resnet_log_helper.log_end_block(output_tensor)

  def log_projection(self, input_tensor, output_tensor):
    if self.model == 'resnet50':
      resnet_log_helper.log_projection(input_tensor, output_tensor)

  def log_conv2d(self, input_tensor, output_tensor, stride_height, stride_width,
                 filters, initializer, use_bias):
    """Log a conv2d call."""
    if self.model == 'resnet50':
      assert stride_height == stride_width, (
          '--ml_perf_compliance_logging does not support convolutions where '
          'the stride height is not equal to the stride width. '
          'stride_height=%d, stride_width=%d' % (stride_height, stride_width))
      if isinstance(initializer, tf.truncated_normal_initializer) or (
          isinstance(initializer, tf.variance_scaling_initializer) and
          initializer.distribution == 'truncated_normal'):
        initializer = tags.TRUNCATED_NORMAL
      elif (isinstance(initializer, tf.glorot_uniform_initializer) or
            initializer is None):
        initializer = 'glorot_uniform'
      resnet_log_helper.log_conv2d(input_tensor, output_tensor, stride_width,
                                   filters, initializer, use_bias)

  def log_batch_norm(self, input_tensor, output_tensor, momentum, epsilon,
                     center, scale, training):
    if self.model == 'resnet50':
      resnet_log_helper.log_batch_norm(input_tensor, output_tensor, momentum,
                                       epsilon, center, scale, training)

  def log_train_epochs(self, num_epochs):
    """Logs all the TRAIN_EPOCHs log lines."""
    num_epochs_int = int(num_epochs)
    for i in range(num_epochs_int):
      # MLPerf allows us to print all the train epochs at once instead of
      # printing them as we do them.
      mlperf_log.resnet_print(key=mlperf_log.TRAIN_EPOCH, value=i)
    if num_epochs_int != num_epochs:
      value = (str(num_epochs_int) +
               ', but this epoch only has {}% of the examples of a normal epoch'
               .format(100 * (num_epochs - num_epochs_int)))
      mlperf_log.resnet_print(key=mlperf_log.TRAIN_EPOCH, value=value)

  def log_input_resize_aspect_preserving(self, height, width, scale_factor):
    assert height == width, (
        '--ml_perf_compliance_logging does not support models with nonsquare '
        'images. Cannot process image with height=%d and width=%d' %
        (height, width))
    self.log(key=tags.INPUT_RESIZE_ASPECT_PRESERVING,
             value={'min': int(height * scale_factor)})


def _empty_fn(*args, **kwargs):
  del args, kwargs


class NullMlPerfLogger(object):
  """A version of `MlPerfLogger` that does not log anything.

  This class has the same interface as `MlPerfLogger`, but does not actually do
  anything. This is used when logging is disabled, which is the default
  behavior.
  """

  def __getattr__(self, item):
    return _empty_fn


# A global singleton logger. By default, it's the null logger but can be
# switched to an MlPerfLogger with `mlperf_logger()`.
logger = NullMlPerfLogger()


@contextlib.contextmanager
def mlperf_logger(use_mlperf_logger, model):
  """Optionally enable the mlperf logger.

  If `use_mlperf_logger` is True, sets the `logger` global variable to an
  instance of MlPerfLogger that will print logs for MLPerf compliance. If
  `use_mlperf_logger` is False, does nothing.

  Args:
    use_mlperf_logger: If True, enables the mlperf logger. If False, this
      function does nothing.
    model: The model that will be logged. Required, because different models
      must log different things for MLPerf compliance.

  Yields:
    Nothing.

  Raises:
    ImportError: If `use_mlperf_logger` is True but the MLPerf compliance
      library cannot be imported
  """
  global logger
  if use_mlperf_logger:
    if not import_successful:
      raise ImportError('Failed to import MLPerf compliance library, which is '
                        'required when --ml_perf_compliance_logging is '
                        'specified. Clone this repo and add this directory '
                        'https://github.com/mlperf/training/tree/master/'
                        'compliance to the PYTHONPATH environmental variable.')
    logger_ = MlPerfLogger(model)
    old_logger = logger
    try:
      logger = logger_
      yield
    finally:
      logger = old_logger
  else:
    yield
