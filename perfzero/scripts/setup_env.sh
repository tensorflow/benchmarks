#!/bin/bash

#export PERFZERO_BENCHMARK_METHODS=unit_test
#export PERFZERO_BENCHMARK_CLASS=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests
export PERFZERO_BENCHMARK_METHODS=benchmark_fake_1gpu_gpuparams
export PERFZERO_BENCHMARK_CLASS=leading_indicators_test.Resnet50Benchmarks
export PERFZERO_PLATFORM_NAME=workstation
export PERFZERO_SYSTEM_NAME=z420
export PERFZERO_TEST_ENV=local
export PERFZERO_PYTHON_PATH=models,benchmarks/scripts/tf_cnn_benchmarks
export PERFZERO_GIT_REPOS="models;https://github.com/tensorflow/models.git,benchmarks;https://github.com/tensorflow/benchmarks.git"
export PERFZERO_OUTPUT_GCS_URL=gs://tf-performance/test-results
export PERFZERO_GCS_DOWNLOADS="cifar10_data;gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/*,gs://tf-perf-imagenet-uswest1/tensorflow/fake_tf_record_data"
export PERFZERO_BIGQUERY_TABLE_NAME="benchmark_results_dev.result"
export PERFZERO_BIGQUERY_PROJECT_NAME="google.com:tensorflow-performance"
