# PerfZero

This libary provides solutions to benchmark Tensorflow performance


# Getting Started


## Setup benchmark environment

The commands below builds the docker image named `temp/tf-gpu` which contains the
libraries needed for benchmark. It also downloads access token from GCS.

This setup script only needs to be executed when we need to create docker image for the
first time, or when we need to update docker image based on the latest
Tensorflow nighly build.

```
python benchmarks/perfzero/lib/setup.py
```


## Run benchmark

Run the command to run benchmark with debug level logging and upload results to
google-cloud-storage and bigquery:

```
nvidia-docker run -it --rm -v $(pwd):/workspace -v /data:/data temp/tf-gpu \
python /workspace/benchmarks/perfzero/lib/benchmark.py \
--output_gcs_url=gs://tf-performance/test-results \
--bigquery_project_name=google.com:tensorflow-performance \
--bigquery_dataset_table_name=benchmark_results_dev.result \
--git_repos=https://github.com/tensorflow/models.git,https://github.com/tensorflow/benchmarks.git \
--gcs_downloads=gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/ \
--python_path=models,benchmarks/scripts/tf_cnn_benchmarks \
--benchmark_methods=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test
--debug
```


## Run all unit tests

```
cd benchmarks/perfzero/lib

python3 -B -m unittest discover -p "*_test.py"
```

## Format python code style

```
find perfzero/lib -name *.py -exec pyformat --in_place {} \;

find perfzero/lib -name *.py -exec gpylint --mode=oss_tensorflow {} \;
```

