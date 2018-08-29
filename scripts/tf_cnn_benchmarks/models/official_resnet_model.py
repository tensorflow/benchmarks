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
"""Import official resnet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datasets
from models import model as model_lib


class ImagenetResnetModel(model_lib.CNNModel):
  """Official resnet models."""

  def __init__(self, resnet_size, version=2, params=None):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      version: 1 or 2 for v1 or v2, respectively.
      params: params passed by BenchmarkCNN.
    """
    default_batch_sizes = {
        50: 128,
        101: 32,
        152: 32
    }
    batch_size = default_batch_sizes.get(resnet_size, 32)
    default_learning_rate = 0.0125 * batch_size / 32
    model_name = 'official_resnet_{}_v{}'.format(resnet_size, version)
    super(ImagenetResnetModel, self).__init__(
        model_name, 224, batch_size, default_learning_rate, params=params)
    self.resnet_size = resnet_size
    self.version = version

  def get_learning_rate(self, global_step, batch_size):
    num_batches_per_epoch = (
        float(datasets.IMAGENET_NUM_TRAIN_IMAGES) / batch_size)
    boundaries = [int(num_batches_per_epoch * x) for x in [30, 60, 80, 90]]
    values = [1, 0.1, 0.01, 0.001, 0.0001]
    adjusted_learning_rate = (
        self.learning_rate / self.default_batch_size * batch_size)
    values = [v * adjusted_learning_rate for v in values]
    return tf.train.piecewise_constant(global_step, boundaries, values)

  def build_network(self, images, phase_train=True, nclass=1001,
                    data_type=tf.float32):
    # pylint: disable=g-import-not-at-top
    try:
      from official.resnet.imagenet_main import ImagenetModel
    except ImportError:
      tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH.')
      raise
    images = tf.cast(images, data_type)
    model_class = ImagenetModel(resnet_size=self.resnet_size,
                                resnet_version=self.version,
                                # The official model dtype seems to be ignored,
                                # as the dtype it uses is the dtype of the input
                                # images. Doesn't hurt to set it though.
                                dtype=data_type)
    logits = model_class(images, phase_train)
    logits = tf.cast(logits, tf.float32)
    return model_lib.BuildNetworkResult(logits=logits, extra_info=None)
