"""Googlenet model configuration.

References:
  Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
  Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich
  Going deeper with convolutions
  arXiv preprint arXiv:1409.4842 (2014)
"""

import model


class GooglenetModel(model.Model):

  def __init__(self):
    super(GooglenetModel, self).__init__('googlenet', 224, 32, 0.005)

  def add_inference(self, cnn):
    def inception_v1(cnn, k, l, m, n, p, q):
      cols = [[('conv', k, 1, 1)], [('conv', l, 1, 1), ('conv', m, 3, 3)],
              [('conv', n, 1, 1), ('conv', p, 5, 5)],
              [('mpool', 3, 3, 1, 1, 'SAME'), ('conv', q, 1, 1)]]
      cnn.inception_module('incept_v1', cols)

    cnn.conv(64, 7, 7, 2, 2)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    cnn.conv(64, 1, 1)
    cnn.conv(192, 3, 3)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 64, 96, 128, 16, 32, 32)
    inception_v1(cnn, 128, 128, 192, 32, 96, 64)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 192, 96, 208, 16, 48, 64)
    inception_v1(cnn, 160, 112, 224, 24, 64, 64)
    inception_v1(cnn, 128, 128, 256, 24, 64, 64)
    inception_v1(cnn, 112, 144, 288, 32, 64, 64)
    inception_v1(cnn, 256, 160, 320, 32, 128, 128)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 256, 160, 320, 32, 128, 128)
    inception_v1(cnn, 384, 192, 384, 48, 128, 128)
    cnn.apool(7, 7, 1, 1, mode='VALID')
    cnn.reshape([-1, 1024])
