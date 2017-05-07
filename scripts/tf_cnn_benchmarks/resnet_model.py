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

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""

from six.moves import xrange  # pylint: disable=redefined-builtin

import model as model_lib


class Resnetv1Model(model_lib.Model):
  """Resnet V1 cnn network configuration."""

  def __init__(self, model, layer_counts):
    defaults = {
        'resnet50': 64,
        'resnet101': 32,
        'resnet152': 32,
    }
    batch_size = defaults.get(model, 32)
    super(Resnetv1Model, self).__init__(model, 224, batch_size, 0.005,
                                        layer_counts)

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET')
    cnn.mpool(3, 3, 2, 2)
    for _ in xrange(self.layer_counts[0]):
      cnn.resnet_bottleneck_v1(256, 64, 1)
    for i in xrange(self.layer_counts[1]):
      stride = 2 if i == 0 else 1
      cnn.resnet_bottleneck_v1(512, 128, stride)
    for i in xrange(self.layer_counts[2]):
      stride = 2 if i == 0 else 1
      cnn.resnet_bottleneck_v1(1024, 256, stride)
    for i in xrange(self.layer_counts[3]):
      stride = 2 if i == 0 else 1
      cnn.resnet_bottleneck_v1(2048, 512, stride)
    cnn.spatial_mean()
