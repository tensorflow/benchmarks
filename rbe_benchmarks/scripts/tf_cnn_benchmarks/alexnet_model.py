"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

import tensorflow as tf
import model


class AlexnetModel(model.Model):
  """Alexnet cnn model."""

  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512, 0.005)

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


class AlexnetCifar10Model(model.Model):
  """Alexnet cnn model for cifar datasets.

  The model architecture follows the one defined in the tensorflow tutorial
  model.

  Reference model: tensorflow/models/tutorials/image/cifar10/cifar10.py
  Paper: http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
  """

  def __init__(self):
    super(AlexnetCifar10Model, self).__init__('alexnet', 32, 128, 0.1)

  def add_inference(self, cnn):
    cnn.conv(64, 5, 5, 1, 1, 'SAME', stddev=5e-2)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    cnn.lrn(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    cnn.conv(64, 5, 5, 1, 1, 'SAME', bias=0.1, stddev=5e-2)
    cnn.lrn(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    shape = cnn.top_layer.get_shape().as_list()
    flat_dim = shape[1] * shape[2] * shape[3]
    cnn.reshape([-1, flat_dim])
    cnn.affine(384, stddev=0.04, bias=0.1)
    cnn.affine(192, stddev=0.04, bias=0.1)

  def get_learning_rate(self, global_step=None, batch_size=None):
    if global_step is None or batch_size is None:
      return self.learning_rate
    num_examples_per_epoch = 50000
    num_epochs_per_decay = 100
    decay_steps = int(num_epochs_per_decay * num_examples_per_epoch /
                      batch_size)
    decay_factor = 0.1
    return tf.train.exponential_decay(
        self.learning_rate, global_step, decay_steps, decay_factor,
        staircase=True)
