# PerfZero

This libary provides solutions to benchmark Tensorflow performance


# Getting Started



## Updating PerfZero configuration

User should specify the PerfZero configuration through environment variables
before setting up and executing benchmark. The full list of environment
variables with example values can be found in `benchmarks/perfzero/scripts/setup_env.sh`.

Edit `setup_env.sh` to specify e.g. the benchmark method before running `source setup_env.sh`.

Here is the full list of environment variables and their documentation:


```
# Benchmark method including its module, class and method name
PERFZERO_BENCHMARK_METHOD

# Name that describes the platform (e.g. aws)
PERFZERO_PLATFORM_NAME

# Name that describes the system hardware (e.g. z420)
PERFZERO_SYSTEM_NAME

# List of paths separated by ',' to be added to python path environment variable
PERFZERO_PYTHON_PATH

# List of git repository url separated by ',' to be checked-out
PERFZERO_GIT_REPOS

# GCS url to upload the benchmark log files
PERFZERO_OUTPUT_GCS_URL

# List of GCS urls separated by ',' to download data
PERFZERO_GCS_DOWNLOADS="gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data,gs://tf-perf-imagenet-uswest1/tensorflow/fake_tf_record_data"

# Name of the bigquery table to upload benchmark result
PERFZERO_BIGQUERY_TABLE_NAME

# Bigquery project id to upload benchmark result
PERFZERO_BIGQUERY_PROJECT_NAME
```


## Setup benchmark environment

The commands below builds the docker image named `temp/tf-gpu` which contains the
libraries needed for benchmark. It also downloads access token from GCS.

This setup script only needs to be executed when we need to create docker image for the
first time, or when we need to update docker image based on the latest
Tensorflow nighly build.

The commands below performs setup operation using the configs specified in `staging/scripts/setup_env.sh`

```
source benchmarks/perfzero/scripts/setup_env.sh

python benchmarks/perfzero/lib/setup.py
```

Alternatively, run the following command:

```
python benchmarks/perfzero/lib/setup.py --config_mode=flags
```


## Run benchmark

The commands below run benchmarks specificed in `staging/scripts/setup_env.sh`


```
source benchmarks/perfzero/scripts/setup_env.sh

nvidia-docker run -it --rm -v $(pwd):/workspace \
-v /data:/data \
-e PERFZERO_BENCHMARK_METHOD \
-e PERFZERO_PLATFORM_NAME \
-e PERFZERO_SYSTEM_NAME \
-e PERFZERO_PYTHON_PATH \
-e PERFZERO_GIT_REPOS \
-e PERFZERO_OUTPUT_GCS_URL \
-e PERFZERO_GCS_DOWNLOADS \
-e PERFZERO_BIGQUERY_TABLE_NAME \
-e PERFZERO_BIGQUERY_PROJECT_NAME \
-e PERFZERO_GCE_NVME_RAID \
temp/tf-gpu \
python /workspace/benchmarks/perfzero/lib/benchmark.py
```

Alterantively, run the following command


```
nvidia-docker run -it --rm -v $(pwd):/workspace -v /data:/data temp/tf-gpu \
python /workspace/benchmarks/perfzero/lib/benchmark.py \
--git_repos=https://github.com/tensorflow/models.git,https://github.com/tensorflow/benchmarks.git \
--gcs_downloads=gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/ \
--output_gcs_url=gs://tf-performance/test-results \
--bigquery_project_name=google.com:tensorflow-performance \
--bigquery_dataset_table_name=benchmark_results_dev.result \
--python_path=models,benchmarks/scripts/tf_cnn_benchmarks \
--benchmark_method=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test \
--config_mode=flags
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

