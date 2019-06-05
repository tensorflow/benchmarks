Table of Contents
=================

   * [Table of Contents](#table-of-contents)
   * [Introduction](#introduction)
   * [Instructions for PerfZero user](#instructions-for-perfzero-user)
      * [Build docker image](#build-docker-image)
      * [Run benchmark](#run-benchmark)
      * [Understand the benchmark execution output](#understand-the-benchmark-execution-output)
         * [Json formatted benchamrk summary](#json-formatted-benchamrk-summary)
         * [Visualize TensorFlow graph etc. using Tensorboard](#visualize-tensorflow-graph-etc-using-tensorboard)
         * [Visualize system metric values over time](#visualize-system-metric-values-over-time)
   * [Instructions for developer who writes benchmark classes](#instructions-for-developer-who-writes-benchmark-classes)
   * [Instructions for PerfZero developer](#instructions-for-perfzero-developer)
   * [Instructions for managing Google Cloud Platform computing instance](#instructions-for-managing-google-cloud-platform-computing-instance)


# Introduction

PerfZero is a benchmark framework for TensorFlow. It intends to address the
following use-cases:

1) For user who wants to execute TensorFlow test to debug performance
regression.

PerfZero makes it easy to execute the pre-defined test by consolidating the
docker image build, GPU driver installation, TensorFlow installation, benchmark
library checkout, data download, system statistics collection, benchmark metrics
collection, profiler data collection and so on into 2 to 3 commands. This allows
developer to focus on investigating the issue rather than setting up the test
environment.

2) For user who wants to track the performance change of TensorFlow for a
variety of setup (e.g. GPU model, cudnn version, TensorFlow version)

The developer can setup periodic job to execute these benchmark methods using
PerfZero. PerfZero will collect the information needed to identify the
benchmark (e.g. GPU model, TensorFlow version, dependent library git hash), get
benchmark execution result (e.g. wall time, accuracy, succeeded or not),
summarize the result in a easy-to-read json string and upload the result to
bigquery table. Using the data in the bigquery table, user can then visualize
the performance change in a dashboard, compare performance between different
setup in a table, and trigger alert on when there is performance regression.


# Instructions for PerfZero user

Here are instructions for users who want to run benchmark method using PerfZero.


## Build docker image

The command below builds the docker image named `perfzero/tensorflow` which contains the
libraries (e.g. TensorFlow) needed for benchmark.

```
python3 benchmarks/perfzero/lib/setup.py
```

Here are a few selected optional flags. Run `python3 setup.py -h` to see
detailed documentation for all supported flags.

1) Use `--dockerfile_path=docker/Dockerfile_ubuntu_1804_tf_v2` to build docker image for TensorFlow v2
2) Use `--tensorflow_pip_spec` to specify the tensorflow pip package name (and optionally version) to be
installed in the docker image, e.g. `--tensorflow_pip_spec=tensorflow==1.12.0`.


## Run benchmark

The command below executes the benchmark method specified by `--benchmark_methods`.

```
export ROOT_DATA_DIR=/data

nvidia-docker run -it --rm -v $(pwd):/workspace -v $ROOT_DATA_DIR:$ROOT_DATA_DIR perfzero/tensorflow \
python3 /workspace/benchmarks/perfzero/lib/benchmark.py \
--gcloud_key_file_url="" \
--git_repos="https://github.com/tensorflow/models.git" \
--python_path=models \
--benchmark_methods=official.resnet.estimator_benchmark.Resnet50EstimatorBenchmarkSynth.benchmark_graph_1_gpu \
--root_data_dir=$ROOT_DATA_DIR
```

`${ROOT_DATA_DIR}` should be the directory which contains the dataset files
required by the benchmark method. If the flag `--data_downloads` is specified,
PerfZero will download files from the specified url to the directory specified
by the flag `--root_data_dir`. Otherwise, user needs to manually download and
move the dataset files into the directory specified by `--root_data_dirs`. The
default `root_data_dir` is `/data`. Some benchmark methods, like the one run in
the sample command above, do not require any dataset files.

Here are a few selected optional flags. Run `python3 benchmark.py -h` to see
detailed documentation for all supported flags.

1) Use `--workspace=unique_workspace_name` if you need to run multiple benchmark
using different workspace setup. One example usecase is that you may want to
test a branch from a pull request without changing your existing workspace.

2) Use `--debug` if you need to see the debug level logging

3) Use `--git_repos="git_url;git_branch;git_hash"` to checkout a git repo with
the specified git_branch at the specified git_hash to the local folder with the
specified folder name. **Note that** the value of the flag `--git_repos` is
wrapped by the quotation mark `"` so that `;` will not be interpreted by the
bash as the end of the command. Specify the flag once for each repository you
want to checkout.

5) Use `--profiler_enabled_time=start_time:end_time` to collect profiler data
during period `[start_time, end_time)` after the benchmark method execution
starts. Skip `end_time` in the flag value to collect data until the end of
benchmark method execution. See [here](#visualize-tensorflow-graph-etc-using-tensorboard)
for instructions on how to use the generated profiler data.


## Understand the benchmark execution output

### Json formatted benchamrk summary

PerfZero outputs a json-formatted summary that provides the information needed
to understand the benchmark result. The summary is printed in the stdout and
in the file `path_to_perfzero/${workspace}/output/${execution_id}/perfzero.log`.

Here is an example output from PerZero. Explanation is provided inline for each
key when the name of the key is not sufficiently self-explanary.

```
 {
  "ml_framework_info": {                         # Summary of the machine learning framework
    "version": "1.13.0-dev20190206",             # Short version. It is tf.__version__ for TensorFlow
    "name": "tensorflow",                        # Machine learning framework name such as PyTorch
    "build_label": "ml_framework_build_label",   # Specified by the flag --ml_framework_build_label
    "build_version": "v1.12.0-7504-g9b32b5742b"  # Long version. It is tf.__git_version__ for TensorFlow
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
    "harness_info": {
      "url": "https://github.com/tensorflow/benchmarks.git",
      "branch": "master",
      "hash": "75d2991b88630dde10ef65aad8082a6d5cd8b5fc"
    },
    "execution_label": "execution_label"      # Specified by the flag --execution_label
  },

  "system_info": {                            # Summary of the resources in the system that is used to execute the benchmark
    "system_name": "system_name",             # Specified by the flag --system_name
    "accelerator_count": 2,                   # Number of GPUs in the system
    "physical_cpu_count": 8,                  # Number of physical cpu cores in the system. Hyper thread CPUs are excluded.
    "logical_cpu_count": 16,                  # Number of logical cpu cores in the system. Hyper thread CPUs are included.
    "cpu_socket_count": 1,                    # Number of cpu socket in the system.
    "platform_name": "platform_name",         # Specified by the flag --platform_name
    "accelerator_model": "Tesla V100-SXM2-16GB",
    "accelerator_driver_version": "410.48",
    "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz"
  },

  "process_info": {                           # Summary of the resources used by the process to execute the benchmark
    "max_rss": 4269047808,                    # maximum physical memory in bytes used by the process
    "max_vms": 39894450176,                   # maximum virtual memory in bytes used by the process
    "max_cpu_percent": 771.1                  # CPU utilization as a percentage. See psutil.Process.cpu_percent() for more information
  },

  "benchmark_result": {                       # Summary of the benchmark execution results. This is pretty much the same data structure defined in test_log.proto.
                                              # Most values are read from test_log.proto which is written by tf.test.Benchmark.report_benchmark() defined in TensorFlow library.

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

### Visualize TensorFlow graph etc. using Tensorboard

When the flag `--profiler_enabled_time=start_time:end_time` is specified, the
profiler data will be collected and stored in
`path_to_perfzero/${workspace}/output/${execution_id}/profiler_data`.

Run `tensorboard --logdir=perfzero/workspace/output/${execution_id}` or `python3
-m tensorboard.main --logdir=perfzero/workspace/output/${execution_id}` to open
Tensorboard server. If PerfZero is executed on a remote machine, run `ssh -L
6006:127.0.0.1:6006 remote_ip` before opening `http://localhost:6006` in your
browser to access the Tensorboard UI.


### Visualize system metric values over time

PerfZero also records a few useful system metrics (e.g. rss, vms) over time in
the file `path_to_perfzero/${workspace}/output/${execution_id}/process_info.log`.
Run `python perfzero/scripts/plot_process_info.py process_info.log` to generate a
pdf showing the value of these metrics over time.


# Instructions for developer who writes benchmark classes

Here are the instructions that developers of benchmark method needs to follow in
order to run benchmark method in PerfZero. See
[estimator_benchmark.py](https://github.com/tensorflow/models/blob/master/official/resnet/estimator_benchmark.py)
for example test code that supports PerfZero.

1) The benchmark class should extend the TensorFlow python class
[tensorflow.test.Benchmark](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py). The benchmark class constructor should have a
constructor with signature `__init__(self, output_dir, data_dir, **kwargs)`.
Below is the usage for each arguments:

- Benchmark method should put all generated files (e.g. logs) in `output_dir` so that PerfZero can
upload these files to Google Cloud Storage when `--output_gcs_url` is specified.

- Benchmark method should read data from `root_data_dir`. For example, the benchmark method can read data from e.g. `${root_data_dir}/cifar-10-binary`

- `**kwargs` is useful to make the benchmark constructor forward compatible when PerfZero provides more named arguments to the benchmark constructor before
  updating the benchmark class.


2) At the end of the benchmark method execution, the method should call [report_benchmark()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py)
with the following parameters:

```
tf.test.Benchmark.report_benchmark(
  iters=num_iteration,          # The number of iterations of the benchmark.

  wall_time=wall_time_sec,      # Total wall time in sec for all iterations.
  metrics = [                   # List of metric entries
    {
      'name': 'accuracy_top_5', # Metric name
      'value': 80,              # Metric value
      'min_value': 90,          # Optional. Minimum acceptable metric value for the benchmark to succeed.
      'max_value': 99           # Optional. Maximum acceptable metric value for the benchmark to succeed.
    },
    {
      'name': 'accuracy_top_1',
      'value': 99.5
    }
  ]
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


Avoid importing `tensorflow` package in any place that requires the `logging`
package because tensorflow package appears to prevent logging from working
properly. Importing `tensorflow` package only in the method that requires it.

Here are the commands to run unit tests and check code style.

```
# Run all unit tests
# This must be executed in directory perfzero/lib
python3 -B -m unittest discover -p "*_test.py"

# Format python code in place
find perfzero/lib -name *.py -exec pyformat --in_place {} \;

# Check python code format and report warning and errors
find perfzero/lib -name *.py -exec gpylint3 {} \;
```

Here is the command to generate table-of-cotents for this README. Run this
command and copy/paste it to the README.md.

```
./perfzero/scripts/generate-readme-header.sh perfzero/README.md
```


# Instructions for managing Google Cloud Platform computing instance

PerfZero aims to make it easy to run and debug TensorFlow which is usually run
with GPU. However, most users do not have dedicated machine with the expensive
hardware. One cost-effective solution is for users to create machine with the
desired hardward on demand in a public cloud when they need to debug TensorFlow.

We provide a script in PerfZero to make it easy to manage computing instance in
Google Cloud Platform. This assumes that you have access to an existing project
in GCP.

Run `python perfzero/lib/cloud_manager.py -h` for list of commands supported
by the script. Run `cloud_manager.py <command> -h` to see detailed documentation
for all supported flags for the specified `command`.

In most cases, user only needs to run the following commands:

```
# Create a new instance that is unique to your username
python perfzero/lib/cloud_manager.py create --project=project_name

# Query the status of the existing instanced created by your and its IP address
python perfzero/lib/cloud_manager.py status --project=project_name

# Stop the instance
python perfzero/lib/cloud_manager.py stop --project=project_name

# Start the instance
python perfzero/lib/cloud_manager.py start --project=project_name

# Delete the instance
python perfzero/lib/cloud_manager.py delete --project=project_name
```


