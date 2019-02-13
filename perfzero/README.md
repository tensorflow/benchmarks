# PerfZero

PerfZero is a benchmark framework for Tensorflow. It intends to provide an
easy-to-use solution for user to run benchmark specified in third-party
libraries such as e.g. tensorflow/models which is necessary to investigate
performance regression issue.

PerfZero provides compact and thorough summary at the end of the benchmark
execution which allows user to compare benchmark execution results between
different setup and over time. Thus users can also run PerfZero periodically in
an automated fasion to collect data and track performance regression for their
in-house machine learning framework and benchmark method.


# Quick start for running benchmark using PerfZero


## Setup benchmark environment

The command below builds the docker image named `temp/tf-gpu` which contains the
libraries needed for benchmark. It also downloads access token from GCS.

This step is needed only when we need to create docker image for the
first time, or when we need to update docker image based on the latest
Tensorflow nighly build.

```
python benchmarks/perfzero/lib/setup.py
```

Optional flag values for setup.py:

1) Use `--dockerfile_path=docker/Dockerfile_ubuntu_1804_tf_v2` to build docker image for tensorflow v2


## Run benchmark

Run the command to run benchmark:

```
nvidia-docker run -it --rm -v $(pwd):/workspace -v /data:/data temp/tf-gpu \
python /workspace/benchmarks/perfzero/lib/benchmark.py \
--git_repos=https://github.com/tensorflow/models.git,https://github.com/tensorflow/benchmarks.git \
--gcs_downloads=gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/ \
--python_path=models,benchmarks/scripts/tf_cnn_benchmarks \
--benchmark_methods=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test
```

Optional flag values for benchmark.py:

1) Remove `--gcs_downloads` if your benchmark does not need to download data

2) Use `--gcs_downloads=gs://tf-perf-imagenet-uswest1/tensorflow/imagenet` to
download data for imagenet benchmark.

3) Use `--workspace=new_workspace_name` if you need to run multiple benchmark
using different workspace setup. One example usecase is that you may want to
test a branch from a pull request without changing your existing workspace.

4) Use `--debug` if you need to see the debug level logging

5) Use `--git_repos=folder_name;git_url;git_branch;git_hash` to checkout a git
repo with the specified git_branch at the specified git_hash to the local folder
with specified folder name.

6) Use `--bigquery_project_name=google.com:tensorflow-performance` and
`--bigquery_dataset_table_name=benchmark_results_dev.result` if you want to
upload results to the specified bigquery project/database/table.

7) Use `--output_gcs_url=gs://tf-performance/test-results` if you want to upload
log files to the specified GCS url.

8) Use additional `--benchmark_methods=path_to_method` to run more than one
benchmark method in one execution.

9) Use `--benchmark_methods=path_to_class.filter:regex_pattern` to run all
methods of the specified benchmark class whose method name matches the given
regex_pattern.


## Understand output


Here is an example output from PerZero. Explanation is provided inline for each
attribute when the attribute name is not sufficiently self-explanary.


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

    "metrics": [                              # This is derived from `extras` in test_log.proto. If the EntryValue is double, then name is the extra's key and value is extra's double value.
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
    "name": "official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests.unit_test",    # Full path to the benchmark method. This is overrideenn by PerfZero.
    "succeeded": true,                        # True iff benchmark method execution finishes successfully and no metric in metrics show succeeded = false
    "wall_time": 14.552583694458008           # It is supposed to be durtion of benchmark execution in seconds. It is -1 if tf.test.Benchmark.report_benchmark() is not called.
  }
}
```



# Developer guide

Here are the example commands that we use to check code quality for PerfZero

## Run all unit tests

```
cd benchmarks/perfzero/lib

python3 -B -m unittest discover -p "*_test.py"
```

## Format python code style

```
# Format python code in place
find perfzero/lib -name *.py -exec pyformat --in_place {} \;

# Check python code format and report warning and errors
find perfzero/lib -name *.py -exec gpylint --mode=oss_tensorflow {} \;
```

