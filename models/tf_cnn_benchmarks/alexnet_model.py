"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

import model


class AlexnetModel(model.Model):
  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512)

  def add_inference(self, cnn):
    # Note: VALID requires padding the images by 3 in width and height
    cnn.conv(64, 11, 11, 4, 4, 'VALID')
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(192, 5, 5)
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(384, 3, 3)
    cnn.conv(384, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.mpool(3, 3, 2, 2)
    cnn.reshape([-1, 256 * 6 * 6])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()
