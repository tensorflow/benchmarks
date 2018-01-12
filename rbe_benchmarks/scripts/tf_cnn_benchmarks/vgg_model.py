"""Vgg model configuration.

Includes multiple models: vgg11, vgg16, vgg19, corresponding to
  model A, D, and E in Table 1 of [1].

References:
[1]  Simonyan, Karen, Andrew Zisserman
     Very Deep Convolutional Networks for Large-Scale Image Recognition
     arXiv:1409.1556 (2014)
"""

from six.moves import xrange  # pylint: disable=redefined-builtin
import model


def _construct_vgg(cnn, num_conv_layers):
  """Build vgg architecture from blocks."""
  assert len(num_conv_layers) == 5
  for _ in xrange(num_conv_layers[0]):
    cnn.conv(64, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[1]):
    cnn.conv(128, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[2]):
    cnn.conv(256, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[3]):
    cnn.conv(512, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[4]):
    cnn.conv(512, 3, 3)
  cnn.mpool(2, 2)
  cnn.reshape([-1, 512 * 7 * 7])
  cnn.affine(4096)
  cnn.dropout()
  cnn.affine(4096)
  cnn.dropout()


class Vgg11Model(model.Model):

  def __init__(self):
    super(Vgg11Model, self).__init__('vgg11', 224, 64, 0.005)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [1, 1, 2, 2, 2])


class Vgg16Model(model.Model):

  def __init__(self):
    super(Vgg16Model, self).__init__('vgg16', 224, 64, 0.005)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [2, 2, 3, 3, 3])


class Vgg19Model(model.Model):

  def __init__(self):
    super(Vgg19Model, self).__init__('vgg19', 224, 64, 0.005)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [2, 2, 4, 4, 4])
