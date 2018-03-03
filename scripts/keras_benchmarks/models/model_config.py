from models import cifar10_cnn_benchmark
from models import gru_benchmark
from models import lstm_benchmark
from models import mnist_mlp_benchmark
from models import resnet50_benchmark
from models import resnet50_benchmark_eager
from models import resnet50_benchmark_subclass
from models import vgg16_benchmark
from models import xception_benchmark

def get_model_config(model_name):
  if model_name == 'cifar10_cnn':
    return cifar10_cnn_benchmark.Cifar10CnnBenchmark()

  if model_name == 'gru':
    return gru_benchmark.GRUBenchmark()

  if model_name == 'lstm':
    return lstm_benchmark.LstmBenchmark()

  if model_name == 'mnist_mlp':
    return mnist_mlp_benchmark.MnistMlpBenchmark()

  if model_name == 'resnet50':
    return resnet50_benchmark.Resnet50Benchmark()

  if model_name == 'vgg16':
    return vgg16_benchmark.VGG16Benchmark()

  if model_name == 'xception':
    return xception_benchmark.XceptionBenchmark()

  if model_name == 'resnet50_eager':
    return resnet50_benchmark_eager.Resnet50Benchmark()

  if model_name == 'resnet50_subclass':
    return resnet50_benchmark_subclass.Resnet50Benchmark()

