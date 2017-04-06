"""Model configurations for CNN benchmarks.
"""

import alexnet_model
import googlenet_model
import inception_model
import lenet_model
import overfeat_model
import resnet_model
import trivial_model
import vgg_model


def get_model_config(model):
  if model == 'vgg11':
    mc = vgg_model.Vgg11Model()
  elif model == 'vgg16':
    mc = vgg_model.Vgg16Model()
  elif model == 'vgg19':
    mc = vgg_model.Vgg19Model()
  elif model == 'lenet':
    mc = lenet_model.Lenet5Model()
  elif model == 'googlenet':
    mc = googlenet_model.GooglenetModel()
  elif model == 'overfeat':
    mc = overfeat_model.OverfeatModel()
  elif model == 'alexnet':
    mc = alexnet_model.AlexnetModel()
  elif model == 'trivial':
    mc = trivial_model.TrivialModel()
  elif model == 'inception3':
    mc = inception_model.Inceptionv3Model()
  elif model == 'inception4':
    mc = inception_model.Inceptionv4Model()
  elif model == 'resnet50':
    mc = resnet_model.Resnetv1Model(model, (2, 3, 5, 3))
  elif model == 'resnet101':
    mc = resnet_model.Resnetv1Model(model, (2, 3, 22, 3))
  elif model == 'resnet152':
    mc = resnet_model.Resnetv1Model(model, (2, 7, 35, 3))
  elif model == 'inception-resnet2':
    mc = inception_model.InceptionResnetv2Model()
  else:
    raise KeyError('Invalid model name \'%s\'' % model)
  return mc

