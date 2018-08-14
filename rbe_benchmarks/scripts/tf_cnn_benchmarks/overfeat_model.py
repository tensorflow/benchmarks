"""Overfeat model configuration.

References:
  OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks
  Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus,
  Yann LeCun, 2014
  http://arxiv.org/abs/1312.6229
"""

import model


class OverfeatModel(model.Model):

  def __init__(self):
    super(OverfeatModel, self).__init__('overfeat', 231, 32, 0.005)

  def add_inference(self, cnn):
    # Note: VALID requires padding the images by 3 in width and height
    cnn.conv(96, 11, 11, 4, 4, mode='VALID')
    cnn.mpool(2, 2)
    cnn.conv(256, 5, 5, 1, 1, mode='VALID')
    cnn.mpool(2, 2)
    cnn.conv(512, 3, 3)
    cnn.conv(1024, 3, 3)
    cnn.conv(1024, 3, 3)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 1024 * 6 * 6])
    cnn.affine(3072)
    cnn.affine(4096)
