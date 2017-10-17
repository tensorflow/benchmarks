from models import mnist_mlp_benchmark
import upload_benchmarks_bq as bq

model = mnist_mlp_benchmark.MnistMlpBenchmark()
model.benchmarkMnistMlp()
bq.upload_metrics_to_bq(model.get_testname(), model.get_totaltime(), model.get_iters(), model.get_sampletype())
