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

import model


class Resnetv1Model(model.Model):
  def __init__(self, model, layer_counts):
    defaults = {
        "resnet50": 64,
        "resnet101": 32,
        "resnet152": 32,
    }
    batch_size = defaults.get(model, 32)
    super(Resnetv1Model, self).__init__(model, 224, batch_size, layer_counts)

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET')
    cnn.mpool(3, 3, 2, 2)
    for _ in xrange(self.layer_counts[0]):
      cnn.resnet_bottleneck_v1(256, 64, 1)
    cnn.resnet_bottleneck_v1(256, 64, 2)
    for _ in xrange(self.layer_counts[1]):
      cnn.resnet_bottleneck_v1(512, 128, 1)
    cnn.resnet_bottleneck_v1(512, 128, 2)
    for _ in xrange(self.layer_counts[2]):
      cnn.resnet_bottleneck_v1(1024, 256, 1)
    cnn.resnet_bottleneck_v1(1024, 256, 2)
    for _ in xrange(self.layer_counts[3]):
      cnn.resnet_bottleneck_v1(2048, 512, 1)
    cnn.spatial_mean()

