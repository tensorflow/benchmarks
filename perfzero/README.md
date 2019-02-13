Table of Contents
=================

   * [Introduction](#introduction)
   * [Instructions for PerfZero user](#instructions-for-perfzero-user)
      * [Build docker image](#build-docker-image)
      * [Run benchmark](#run-benchmark)
      * [Understand benchmark summary](#understand-benchmark-summary)
   * [Instructions for benchmark developer](#instructions-for-benchmark-developer)
   * [Instructions for PerfZero developer](#instructions-for-perfzero-developer)

# Introduction

PerfZero is a benchmark framework for Tensorflow. It intends to address the
following use-cases:

1) For user who wants to execute Tensorflow test to debug performance
regression.

PerfZero makes it easy to execute the pre-defined test by consolidating the
docker image build, GPU driver installation, Tensorflow installation, benchmark
library checkout, data download, system statistics collection, benchmark
metrics collection and so on into 2 to 3 commands. This allows developer to focus
on investigating the issue rather than setting up the test environment.

2) For user who wants to track the performance change of Tensorflow for a
variety of setup (e.g. GPU model, cudnn version, Tensorflow version)

The developer can setup periodic job to execute these benchmark methods using
PerfZero. PerfZero will collect the information needed to identify the
benchmark (e.g. GPU model, Tensorflow version, dependent library git hash), get
benchmark execution result (e.g. wall time, accuracy, succeeded or not),
summarize the result in a easy-to-read json string and upload the result to
bigquery table. Using the data in the bigquery table, user can then visualize
the performance change in a dashboard, compare performance between different
setup in a table, and trigger alert on when there is performance regression.




# Instructions for PerfZero user

Here are instructions for users who want to run benchmark method using PerfZero.


## Build docker image

The command below builds the docker image named `temp/tf-gpu` which contains the
libraries (e.g. Tensorflow) needed for benchmark.

```
python3 benchmarks/perfzero/lib/setup.py
```

Optional flag values:

1) Use `--dockerfile_path=docker/Dockerfile_ubuntu_1804_tf_v2` to build docker image for Tensorflow v2


## Run benchmark

The command below executes the benchmark method specified by `--benchmark_methods`:

```
nvidia-docker run -it --rm -v $(pwd):/workspace -v /data:/data temp/tf-gpu \
python3 /workspace/benchmarks/perfzero/lib/benchmark.py \
--git_repos=https://github.com/tensorflow/models.git \
--python_path=models \
--benchmark_methods=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test
```

Note that if there is data required by the benchmark method, make sure the data
is available at the path expected by the benchmark method. For example,
benchmark methods defined in
[tensorflow/models](https://github.com/tensorflow/models) expects data to be
under the path `/data`. You can manually copy the data to `${path}` and then run
nvidia-docker with argument `-v ${path}:/data`. This allows *benchmark.py* to get
the data at the path `/data` when it is executed insider docker.

Alternatively, upload the data to Google Cloud Storage and provide the url to
the argument `--gcs_downloads`, which lets *benchmark.py* download the data from
GCS to the path `/data`.

Here are a few commonly used optional flag values. Run `python3 benchmarkpy.py --help` for detail.

1) Use `--gcs_downloads=gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data`
to download data for imagenet benchmark defined in tensorflow/models (if you
have the permission).

2) Use `--workspace=unique_workspace_name` if you need to run multiple benchmark
using different workspace setup. One example usecase is that you may want to
test a branch from a pull request without changing your existing workspace.

3) Use `--debug` if you need to see the debug level logging

4) Use `--git_repos=git_url;git_branch;git_hash` to checkout a git
repo with the specified git_branch at the specified git_hash to the local folder
with specified folder name.


## Understand benchmark summary

PerfZero outputs a json-formatted summary that provides the information needed
to understand the benchmark result. The summary is printed in the stdout and
logged in file under the directory
`path_to_perfzero/${workspace}/output/${execution_id}`. It is also uploaded to
bigquery in Google Cloud when `--bigquery_dataset_table_name` is specified.

Here is an example output from PerZero. Explanation is provided inline for each
key when the name of the key is not sufficiently self-explanary.

```
 {
  "ml_framework_info": {                         # Summary of the machine learning framework
    "version": "1.13.0-dev20190206",             # Short version. It is tf.__version__ for Tensorflow
    "name": "tensorflow",                        # Machine learning framework name such as PyTorch
    "build_label": "ml_framework_build_label",   # Specified by the flag --ml_framework_build_label
    "build_version": "v1.12.0-7504-g9b32b5742b"  # Long version. It is tf.__git_version__ for Tensorflow
  },
  "execution_timestamp": 1550040322.8991697,     # Timestamp when the benchmark is executed
  "execution_id": "2019-02-13-06-45-22-899128",  # A string that uniquely identify this benchmark execution

  "benchmark_info": {                            # Summary of the benchmark framework setup
    "output_url": "gs://tf-performance/test-results/2019-02-13-06-45-22-899128/",     # Google storage url that contains the log file from this benchmark execution
    "has_exception": false,
    "site_package_info": {
      "models": {
        "branch": "master",
        "url": "https://github.com/tensorflow/models.git",
        "hash": "f788046ca876a8820e05b0b48c1fc2e16b0955bc"
      },
      "benchmarks": {
        "branch": "master",
        "url": "https://github.com/tensorflow/benchmarks.git",
        "hash": "af9e0ef36fc6867d9b63ebccc11f229375cd6a31"
      }
    },
    "harness_name": "perfzero",
    "execution_label": "execution_label"      # Specified by the flag --execution_label
  },

  "system_info": {                            # Summary of the system that is used to execute the benchmark
    "system_name": "system_name",             # Specified by the flag --system_name
    "accelerator_count": 2,
    "cpu_core_count": 8,
    "platform_name": "platform_name",         # Specified by the flag --platform_name
    "cpu_socket_count": 1,
    "accelerator_model": "Tesla V100-SXM2-16GB",
    "accelerator_driver_version": "410.48",
    "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz"
  },

  "benchmark_result": {                       # Summary of the benchmark execution results. This is pretty much the same data structure defined in test_log.proto.
                                              # Most values are read from test_log.proto which is written by tf.test.Benchmark.report_benchmark() defined in Tensorflow library.

    "metrics": [                              # This is derived from `extras` [test_log.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto) 
                                              # which is written by report_benchmark().
                                              # If the EntryValue is double, then name is the extra's key and value is extra's double value.
                                              # If the EntryValue is string, then name is the extra's key. The string value will be a json formated string whose keys
                                              # include `value`, `succeeded` and `description`. Benchmark method can provide arbitrary metric key/value pairs here.
      {
        "name": "accuracy_top_5",
        "value": 0.7558000087738037
      },
      {
        "name": "accuracy_top_1",
        "value": 0.2639999985694885
      }
    ],
    "name": "official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test",    # Full path to the benchmark method, i.e. module_path.class_name.method_name
    "succeeded": true,                        # True iff benchmark method execution finishes without exception and no metric in metrics show succeeded = false
    "wall_time": 14.552583694458008           # The value is determined by tf.test.Benchmark.report_benchmark() called by the benchmark method. It is -1 if report_benchmark() is not called.
  }
}
```

# Instructions for benchmark developer

Here are the instructions that developers of benchmark method needs to follow in
order to run benchmark method in PerfZero.

1) The benchmark class should extend the Tensorflow python class
[tensorflow.test.Benchmark](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py). The benchmark class constructor should have a
constructor with signature `__init__(self, output_dir)`. Benchmark method
should put all generated files (e.g. logs) in `output_dir` so that PerfZero can
upload these files to Google Cloud Storage when `--output_gcs_url` is specified.
See [EstimatorCifar10BenchmarkTests](https://github.com/tensorflow/models/blob/master/official/resnet/estimator_cifar_benchmark.py) for example.

2) At the end of the benchmark method execution, the method should call [report_benchmark()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py) 
with the following parameters:

```
tf.test.Benchmark.report_benchmark(
  iters=num_iteration,         # The number of iterations of the benchmark.

  wall_time=wall_time_sec,     # Total wall time in sec for all iterations.

  extras={                     # key/value pairs for arbitrary metrics where key is the name of the metric and value can be either string or double.

      'accuracy': 99.5,        # If the value is double, it is used as the value of the metric.

      'accuracy_top_5': {      # If the value is string, it should be a json-formatted string with the following fields.
        'value': 80,           # This is used as the value of the metric
        'succeeded': True      # Optional. Whether the metric passes developer-defined requirement. This is used by PerfZero to report whether benchmark has succeeded.
        'min_value': 80        # Optional. Lower bound for the metric value to succeed. Used to tell user why the benchmark has failed.
        'min_value': 100       # Optional. Upper bound for the metric value to succeed. Used to tell user why the benchmark has failed.
      }
  }
)
```

This format allows PerfZero to specify whether the benchmark has succeeded
(e.g.  for convergence test) in its summary based on the logic determined by
the benchmark developer.


3) Include dependent libraries in `--git_repos` and `--python_path`. These
libraries will be checked-out in the directory
`path_to_perfzero/workspace/site-packages` by default. Developer can edit these
libraries directly and execute benchmark with the local change.


# Instructions for PerfZero developer

Here are the instructions for developers who want to contribute code to PerfZero

```
# Run all unit tests
# This must be executed in directory perfzero/lib
python3 -B -m unittest discover -p "*_test.py"

# Format python code in place
find perfzero/lib -name *.py -exec pyformat --in_place {} \;

# Check python code format and report warning and errors
find perfzero/lib -name *.py -exec pylint {} \;
```

Here is the command to generate table-of-cotents for this README. Run this
command and copy/paste it to the README.md.

```
./perfzero/scripts/generate-readme-header.sh perfzero/README.md
```

