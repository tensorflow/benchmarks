"""Inception model configuration.

Includes multiple models: inception3, inception4, inception-resnet2.

References:
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
  Inception-v4, Inception-ResNet and the Impact of Residual Connections on
  Learning

  Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
  Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
  Going Deeper with Convolutions
  http://arxiv.org/pdf/1409.4842v1.pdf

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna
  Rethinking the Inception Architecture for Computer Vision
  arXiv preprint arXiv:1512.00567 (2015)

  Inception v3 model: http://arxiv.org/abs/1512.00567

  Inception v4 and Resnet V2 architectures: http://arxiv.org/abs/1602.07261
"""

import model


class Inceptionv3Model(model.Model):
  def __init__(self):
    super(Inceptionv3Model, self).__init__('inception3', 299, 32)

  def add_inference(self, cnn):
    def inception_v3_a(cnn, n):
      cols = [[('conv', 64, 1, 1)], [('conv', 48, 1, 1), ('conv', 64, 5, 5)],
              [('conv', 64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)],
              [('apool', 3, 3, 1, 1, 'SAME'), ('conv', n, 1, 1)]]
      cnn.inception_module('incept_v3_a', cols)

    def inception_v3_b(cnn):
      cols = [[('conv', 384, 3, 3, 2, 2, 'VALID')],
              [('conv', 64, 1, 1),
               ('conv', 96, 3, 3),
               ('conv', 96, 3, 3, 2, 2, 'VALID')],
              [('mpool', 3, 3, 2, 2, 'VALID')]]
      cnn.inception_module('incept_v3_b', cols)

    def inception_v3_c(cnn, n):
      cols = [[('conv', 192, 1, 1)],
              [('conv', n, 1, 1), ('conv', n, 1, 7), ('conv', 192, 7, 1)],
              [('conv', n, 1, 1), ('conv', n, 7, 1), ('conv', n, 1, 7),
               ('conv', n, 7, 1), ('conv', 192, 1, 7)],
              [('apool', 3, 3, 1, 1, 'SAME'), ('conv', 192, 1, 1)]]
      cnn.inception_module('incept_v3_c', cols)

    def inception_v3_d(cnn):
      cols = [[('conv', 192, 1, 1), ('conv', 320, 3, 3, 2, 2, 'VALID')],
              [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 192, 7, 1),
               ('conv', 192, 3, 3, 2, 2, 'VALID')],
              [('mpool', 3, 3, 2, 2, 'VALID')]]
      cnn.inception_module('incept_v3_d', cols)

    def inception_v3_e(cnn, pooltype):
      cols = [[('conv', 320, 1, 1)], [('conv', 384, 1, 1), ('conv', 384, 1, 3)],
              [('share',), ('conv', 384, 3, 1)],
              [('conv', 448, 1, 1), ('conv', 384, 3, 3), ('conv', 384, 1, 3)],
              [('share',), ('share',), ('conv', 384, 3, 1)],
              [('mpool' if pooltype == 'max' else 'apool', 3, 3, 1, 1, 'SAME'),
               ('conv', 192, 1, 1)]]
      cnn.inception_module('incept_v3_e', cols)

    # TODO: This does not include the extra 'arm' that forks off
    #         from before the 3rd-last module (the arm is designed
    #         to speed up training in the early stages).
    cnn.use_batch_norm = True
    cnn.conv(32, 3, 3, 2, 2, mode='VALID')
    cnn.conv(32, 3, 3, 1, 1, mode='VALID')
    cnn.conv(64, 3, 3, 1, 1, mode='SAME')
    cnn.mpool(3, 3, 2, 2, mode='VALID')
    cnn.conv(80, 1, 1, 1, 1, mode='VALID')
    cnn.conv(192, 3, 3, 1, 1, mode='VALID')
    cnn.mpool(3, 3, 2, 2, 'VALID')
    inception_v3_a(cnn, 32)
    inception_v3_a(cnn, 64)
    inception_v3_a(cnn, 64)
    inception_v3_b(cnn)
    inception_v3_c(cnn, 128)
    inception_v3_c(cnn, 160)
    inception_v3_c(cnn, 160)
    inception_v3_c(cnn, 192)
    inception_v3_d(cnn)
    inception_v3_e(cnn, 'avg')
    inception_v3_e(cnn, 'max')
    cnn.apool(8, 8, 1, 1, 'VALID')
    cnn.reshape([-1, 2048])


# Stem functions
def inception_v4_sa(cnn):
  cols = [[('mpool', 3, 3, 2, 2, 'VALID')], [('conv', 96, 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_sa', cols)


def inception_v4_sb(cnn):
  cols = [[('conv', 64, 1, 1), ('conv', 96, 3, 3, 1, 1, 'VALID')],
          [('conv', 64, 1, 1), ('conv', 64, 7, 1), ('conv', 64, 1, 7),
           ('conv', 96, 3, 3, 1, 1, 'VALID')]]
  cnn.inception_module('incept_v4_sb', cols)


def inception_v4_sc(cnn):
  cols = [[('conv', 192, 3, 3, 2, 2, 'VALID')],
          [('mpool', 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_sc', cols)


# Reduction functions
def inception_v4_ra(cnn, k, l, m, n):
  cols = [
      [('mpool', 3, 3, 2, 2, 'VALID')], [('conv', n, 3, 3, 2, 2, 'VALID')],
      [('conv', k, 1, 1), ('conv', l, 3, 3), ('conv', m, 3, 3, 2, 2, 'VALID')]
  ]
  cnn.inception_module('incept_v4_ra', cols)


def inception_v4_rb(cnn):
  cols = [[('mpool', 3, 3, 2, 2, 'VALID')],
          [('conv', 192, 1, 1), ('conv', 192, 3, 3, 2, 2, 'VALID')],
          [('conv', 256, 1, 1), ('conv', 256, 1, 7), ('conv', 320, 7, 1),
           ('conv', 320, 3, 3, 2, 2, 'VALID')]]
  cnn.inception_module('incept_v4_rb', cols)


def inception_resnet_v2_rb(cnn):
  cols = [
      [('mpool', 3, 3, 2, 2, 'VALID')],
      # TODO: These match the paper but don't match up with the following layer
      # [('conv', 256, 1, 1), ('conv', 384, 3, 3, 2, 2, 'VALID')],
      # [('conv', 256, 1, 1), ('conv', 288, 3, 3, 2, 2, 'VALID')],
      # [('conv', 256, 1, 1), ('conv', 288, 3, 3),
      #        ('conv', 320, 3, 3, 2, 2, 'VALID')]]
      # TODO: These match Facebook's Torch implem
      [('conv', 256, 1, 1), ('conv', 384, 3, 3, 2, 2, 'VALID')],
      [('conv', 256, 1, 1), ('conv', 256, 3, 3, 2, 2, 'VALID')],
      [('conv', 256, 1, 1), ('conv', 256, 3, 3), ('conv', 256, 3, 3, 2, 2,
                                                  'VALID')]
  ]
  cnn.inception_module('incept_resnet_v2_rb', cols)


class Inceptionv4Model(model.Model):
  def __init__(self):
    # TODO: check the default batch size for this model.
    super(Inceptionv4Model, self).__init__('inception4', 299, 32)

  def add_inference(self, cnn):
    def inception_v4_a(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 96, 1, 1)],
              [('conv', 96, 1, 1)], [('conv', 64, 1, 1), ('conv', 96, 3, 3)],
              [('conv', 64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)]]
      cnn.inception_module('incept_v4_a', cols)

    def inception_v4_b(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 128, 1, 1)],
              [('conv', 384, 1, 1)],
              [('conv', 192, 1, 1), ('conv', 224, 1, 7), ('conv', 256, 7, 1)],
              [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 224, 7, 1),
               ('conv', 224, 1, 7), ('conv', 256, 7, 1)]]
      cnn.inception_module('incept_v4_b', cols)

    def inception_v4_c(cnn):
      cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 256, 1, 1)],
              [('conv', 256, 1, 1)], [('conv', 384, 1, 1), ('conv', 256, 1, 3)],
              [('share',), ('conv', 256, 3, 1)],
              [('conv', 384, 1, 1), ('conv', 448, 1, 3), ('conv', 512, 3, 1),
               ('conv', 256, 3, 1)], [('share',), ('share',), ('share',),
                                      ('conv', 256, 1, 3)]]
      cnn.inception_module('incept_v4_c', cols)

    cnn.use_batch_norm = True
    cnn.conv(32, 3, 3, 2, 2, mode='VALID')
    cnn.conv(32, 3, 3, 1, 1, mode='VALID')
    cnn.conv(64, 3, 3)
    inception_v4_sa(cnn)
    inception_v4_sb(cnn)
    inception_v4_sc(cnn)
    for _ in xrange(4):
      inception_v4_a(cnn)
    inception_v4_ra(cnn, 192, 224, 256, 384)
    for _ in xrange(7):
      inception_v4_b(cnn)
    inception_v4_rb(cnn)
    for _ in xrange(3):
      inception_v4_c(cnn)
    cnn.spatial_mean()
    cnn.dropout(0.8)


class InceptionResnetv2Model(model.Model):
  def __init__(self):
    # TODO: check the default batch size for this model.
    super(InceptionResnetv2Model, self).__init__('inception-resnet2', 299, 32)

  def add_inference(self, cnn):
    def inception_resnet_v2_a(cnn):
      cols = [[('conv', 32, 1, 1)], [('conv', 32, 1, 1), ('conv', 32, 3, 3)],
              [('conv', 32, 1, 1), ('conv', 48, 3, 3), ('conv', 64, 3, 3)]]
      cnn.inception_module('incept_resnet_v2_a', cols)

    def inception_resnet_v2_b(cnn):
      cols = [[('conv', 192, 1, 1)],
              [('conv', 128, 1, 1), ('conv', 160, 1, 7), ('conv', 192, 7, 1)]]
      cnn.inception_module('incept_resnet_v2_b', cols)

    def inception_resnet_v2_c(cnn):
      cols = [[('conv', 192, 1, 1)],
              [('conv', 192, 1, 1), ('conv', 224, 1, 3), ('conv', 256, 3, 1)]]
      cnn.inception_module('incept_resnet_v2_c', cols)

    cnn.use_batch_norm = True
    residual_scale = 0.2
    cnn.conv(32, 3, 3, 2, 2, mode='VALID')
    cnn.conv(32, 3, 3, 1, 1, mode='VALID')
    cnn.conv(64, 3, 3)
    inception_v4_sa(cnn)
    inception_v4_sb(cnn)
    inception_v4_sc(cnn)
    for _ in xrange(5):
      cnn.residual(384, inception_resnet_v2_a, scale=residual_scale)
    inception_v4_ra(cnn, 256, 256, 384, 384)
    for _ in xrange(10):
      # TODO: This was 1154 in the paper, but then the layers don't match up
      #         One Caffe model online appears to use 1088
      #         Facebook's Torch implem uses 1152
      cnn.residual(1152, inception_resnet_v2_b, scale=residual_scale)
    inception_resnet_v2_rb(cnn)
    for _ in xrange(5):
      # TODO: This was 2048 in the paper, but then the layers don't match up
      #         One Caffe model online appears to use 2080
      #         Facebook's Torch implem uses 2048 but modifies the
      #         preceding reduction net so that it matches
      # cnn.residual(2144, inception_resnet_v2_c, scale=residual_scale)
      cnn.residual(2048, inception_resnet_v2_c, scale=residual_scale)
    cnn.spatial_mean()
    cnn.dropout(0.8)
