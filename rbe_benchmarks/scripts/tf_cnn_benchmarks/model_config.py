"""Model configurations for CNN benchmarks.
"""

import alexnet_model
import densenet_model
import googlenet_model
import inception_model
import lenet_model
import overfeat_model
import resnet_model
import trivial_model
import vgg_model


def get_model_config(model, dataset):
  """Map model name to model network configuration."""
  if 'cifar10' == dataset.name:
    return get_cifar10_model_config(model)
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
  elif model == 'resnet50' or model == 'resnet50_v2':
    mc = resnet_model.ResnetModel(model, (3, 4, 6, 3))
  elif model == 'resnet101' or model == 'resnet101_v2':
    mc = resnet_model.ResnetModel(model, (3, 4, 23, 3))
  elif model == 'resnet152' or model == 'resnet152_v2':
    mc = resnet_model.ResnetModel(model, (3, 8, 36, 3))
  else:
    raise KeyError('Invalid model name \'%s\' for dataset \'%s\'' %
                   (model, dataset.name))
  return mc


def get_cifar10_model_config(model):
  """Map model name to model network configuration for cifar10 dataset."""
  if model == 'alexnet':
    mc = alexnet_model.AlexnetCifar10Model()
  elif model == 'resnet20' or model == 'resnet20_v2':
    mc = resnet_model.ResnetCifar10Model(model, (3, 3, 3))
  elif model == 'resnet32' or model == 'resnet32_v2':
    mc = resnet_model.ResnetCifar10Model(model, (5, 5, 5))
  elif model == 'resnet44' or model == 'resnet44_v2':
    mc = resnet_model.ResnetCifar10Model(model, (7, 7, 7))
  elif model == 'resnet56' or model == 'resnet56_v2':
    mc = resnet_model.ResnetCifar10Model(model, (9, 9, 9))
  elif model == 'resnet110' or model == 'resnet110_v2':
    mc = resnet_model.ResnetCifar10Model(model, (18, 18, 18))
  elif model == 'densenet40_k12':
    mc = densenet_model.DensenetCifar10Model(model, (12, 12, 12), 12)
  elif model == 'densenet100_k12':
    mc = densenet_model.DensenetCifar10Model(model, (32, 32, 32), 12)
  elif model == 'densenet100_k24':
    mc = densenet_model.DensenetCifar10Model(model, (32, 32, 32), 24)
  else:
    raise KeyError('Invalid model name \'%s\' for Cifar10 DataSet.' % model)
  return mc
