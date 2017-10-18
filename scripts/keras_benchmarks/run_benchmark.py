from models import mnist_mlp_benchmark
import upload_benchmarks_bq as bq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--keras_backend', default="tensorflow",
                    help='Keras backend being used to run benchmarks.')

args = parser.parse_args()
print(args.keras_backend)


model = mnist_mlp_benchmark.MnistMlpBenchmark()
model.benchmarkMnistMlp()
bq.upload_metrics_to_bq(model.get_testname(), model.get_totaltime(), model.get_iters(), model.get_sampletype())
