""" Main entry point for running benchmarks with different Keras backends."""

from models import mnist_mlp_benchmark
from models import cifar10_cnn_benchmark
from models import mnist_irnn_benchmark
from models import lstm_text_generation_benchmark
import upload_benchmarks_bq as bq
import argparse
import keras


parser = argparse.ArgumentParser()
parser.add_argument('--keras_backend', default="tensorflow",
                    help='Keras backend being used to run benchmarks.')

parser.add_argument('--cpu_num_cores',
                    help='')
parser.add_argument('--cpu_memory',
                    help='')
parser.add_argument('--cpu_memory_info',
                    help='')

parser.add_argument('--platform_type',
                    help='')
parser.add_argument('--platform_machine_type',
                    help='')

parser.add_argument('--gpu_count',
                    help='')
parser.add_argument('--gpu_platform',
                    help='')

args = parser.parse_args()

if str(args.keras_backend) == "tensorflow":
    import tensorflow as tf
if str(args.keras_backend) == "theano":
    import theano
if str(args.keras_backend) == "cntk":
    import cntk


def get_backend_version(backend_type):
    if str(backend_type) == "tensorflow":
        return tf.__version__
    if str(backend_type) == "theano":
        return theano.__version__
    if str(backend_type) == "cntk":
        return cntk.__version__
    return "undefined"


def upload_metrics(current_model):
    bq.upload_metrics_to_bq(test_name=current_model.get_testname(),
                            total_time=current_model.get_totaltime(),
                            epochs=current_model.get_iters(),
                            batch_size=current_model.get_batch_size(),
                            backend_type=args.keras_backend,
                            backend_version=get_backend_version(args.keras_backend),
                            cpu_num_cores=args.cpu_num_cores,
                            cpu_memory=args.cpu_memory,
                            cpu_memory_info=args.cpu_memory_info,
                            gpu_count=args.gpu_count,
                            gpu_platform=args.gpu_platform,
                            platform_type=args.platform_type,
                            platform_machine_type=args.platform_machine_type,
                            keras_version=keras.__version__,
                            sample_type=current_model.get_sampletype())


#TODO(anjalisridhar): instantiate models in a loop to avoid calling bq functions repeatedly
#TODO(anjalisridhar): use a dict to pass cpu info etc to the upload function

# MNIST MLP
model = mnist_mlp_benchmark.MnistMlpBenchmark()
model.run_benchmark(args.keras_backend, args.gpu_count)
upload_metrics(model)

# CIFAR10 CNN
model = cifar10_cnn_benchmark.Cifar10CnnBenchmark()
model.run_benchmark(args.keras_backend, args.gpu_count)
upload_metrics(model)

# MNIST RNN
model = mnist_irnn_benchmark.MnistIrnnBenchmark()
model.run_benchmark(args.keras_backend, args.gpu_count)
upload_metrics(model)

# LSTM
model = lstm_text_generation_benchmark.LstmTextGenBenchmark()
model.run_benchmark(args.keras_backend, args.gpu_count)
upload_metrics(model)
