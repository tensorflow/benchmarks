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

"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import os

import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import preprocessing


IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000


def create_dataset(data_dir, data_name):
  """Create a Dataset instance based on data_dir and data_name."""
  supported_datasets = {
      'synthetic': SyntheticData,
      'imagenet': ImagenetData,
      'cifar10': Cifar10Data,
  }
  if not data_dir:
    data_name = 'synthetic'

  if data_name is None:
    for supported_name in supported_datasets:
      if supported_name in data_dir:
        data_name = supported_name
        break

  if data_name is None:
    raise ValueError('Could not identify name of dataset. '
                     'Please specify with --data_name option.')

  if data_name not in supported_datasets:
    raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
        [key for key in sorted(supported_datasets.keys())]))

  return supported_datasets[data_name](data_dir)


class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self, name, height=None, width=None, depth=None, data_dir=None,
               queue_runner_required=False, num_classes=1000):
    self.name = name
    self.height = height
    self.width = width
    self.depth = depth or 3

    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required
    self._num_classes = num_classes

  def tf_record_pattern(self, subset):
    return os.path.join(self.data_dir, '%s-*-of-*' % subset)

  def reader(self):
    return tf.TFRecordReader()

  @property
  def num_classes(self):
    return self._num_classes

  @num_classes.setter
  def num_classes(self, val):
    self._num_classes = val

  @abstractmethod
  def num_examples_per_epoch(self, subset):
    pass

  def __str__(self):
    return self.name

  def get_image_preprocessor(self):
    return None

  def queue_runner_required(self):
    return self._queue_runner_required

  def use_synthetic_gpu_images(self):
    return False


class ImagenetData(Dataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir)

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return IMAGENET_NUM_TRAIN_IMAGES
    elif subset == 'validation':
      return IMAGENET_NUM_VAL_IMAGES
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.RecordInputImagePreprocessor


class SyntheticData(Dataset):
  """Configuration for synthetic dataset."""

  def __init__(self, unused_data_dir):
    super(SyntheticData, self).__init__('synthetic')

  def get_image_preprocessor(self):
    return preprocessing.SyntheticImagePreprocessor

  def use_synthetic_gpu_images(self):
    return True


class Cifar10Data(Dataset):
  """Configuration for cifar 10 dataset.

  It will mount all the input images to memory.
  """

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(Cifar10Data, self).__init__('cifar10', 32, 32, data_dir=data_dir,
                                      queue_runner_required=True,
                                      num_classes=10)

  def read_data_files(self, subset='train'):
    """Reads from data file and return images and labels in a numpy array."""
    if subset == 'train':
      filenames = [os.path.join(self.data_dir, 'data_batch_%d' % i)
                   for i in xrange(1, 6)]
    elif subset == 'validation':
      filenames = [os.path.join(self.data_dir, 'test_batch')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    inputs = []
    for filename in filenames:
      with gfile.Open(filename, 'r') as f:
        inputs.append(cPickle.load(f))
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    all_images = np.concatenate(
        [each_input['data'] for each_input in inputs]).astype(np.float32)
    all_labels = np.concatenate(
        [each_input['labels'] for each_input in inputs])
    return all_images, all_labels

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 50000
    elif subset == 'validation':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.Cifar10ImagePreprocessor
