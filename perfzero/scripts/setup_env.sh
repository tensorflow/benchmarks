#!/bin/bash

export ROGUE_TEST_METHODS=resnet56_1_gpu
export ROGUE_TEST_CLASS=official.resnet.estimator_cifar_benchmark.EstimatorCifar10BenchmarkTests
export ROGUE_PYTHON_PATH=models
export ROGUE_GIT_REPOS="models;https://github.com/tensorflow/models.git"
export ROGUE_CODE_DIR=gs://tf-performance/test-results
export ROGUE_PLATFORM=workstation
export ROGUE_PLATFORM_TYPE=z420
export ROGUE_TEST_ENV=local
export ROGUE_REPORT_PROJECT=LOCAL
export ROGUE_GCS_DOWNLOADS="cifar10_data;gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/*"

