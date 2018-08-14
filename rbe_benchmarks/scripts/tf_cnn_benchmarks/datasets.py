"""Benchmark dataset utilities.
"""

from abc import abstractmethod
import cPickle
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import preprocessing


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
               queue_runner_required=False):
    self.name = name
    self.height = height
    self.width = width
    self.depth = depth or 3
    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required

  def tf_record_pattern(self, subset):
    return os.path.join(self.data_dir, '%s-*-of-*' % subset)

  def reader(self):
    return tf.TFRecordReader()

  @abstractmethod
  def num_classes(self):
    pass

  @abstractmethod
  def num_examples_per_epoch(self, subset):
    pass

  def __str__(self):
    return self.name

  def get_image_preprocessor(self):
    return None

  def queue_runner_required(self):
    return self._queue_runner_required


class ImagenetData(Dataset):
  """Configuration for Imagenet dataset."""

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir)

  def num_classes(self):
    return 1000

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 1281167
    elif subset == 'validation':
      return 50000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.RecordInputImagePreprocessor


class SyntheticData(Dataset):
  """Configuration for synthetic dataset."""

  def __init__(self, unused_data_dir):
    super(SyntheticData, self).__init__('synthetic')

  def num_classes(self):
    return 1000


class Cifar10Data(Dataset):
  """Configuration for cifar 10 dataset.

  It will mount all the input images to memory.
  """

  def __init__(self, data_dir=None):
    if data_dir is None:
      raise ValueError('Data directory not specified')
    super(Cifar10Data, self).__init__('cifar10', 32, 32, data_dir=data_dir,
                                      queue_runner_required=True)

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

  def num_classes(self):
    return 10

  def num_examples_per_epoch(self, subset='train'):
    if subset == 'train':
      return 50000
    elif subset == 'validation':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_image_preprocessor(self):
    return preprocessing.Cifar10ImagePreprocessor
