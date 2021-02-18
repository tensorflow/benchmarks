#!/bin/bash

set -e
set -x

git clone https://github.com/tensorflow/benchmarks.git
cd benchmarks

tf_spec="tf-nightly-gpu==2.5.0.dev20210212"

benchmark_methods=official.benchmark.keras_cifar_benchmark.Resnet56KerasBenchmarkReal.benchmark_1_gpu_no_dist_strat

function run_benchmark() {
  docker_name=$1
  label=$2

  perfzero_pwd=`pwd`

  nvidia-docker run \
    -v ${perfzero_pwd}:/workspace \
    -v /data:/data \
    -e PERFZERO_RUN_TAGS= \
    -e PERFZERO_TRACKING_ID= \
    -e PERFZERO_COMMIT_LABEL= \
    -e PERFZERO_EXECUTION_BRANCH=master \
    ${docker_name} \
    python3 /workspace/perfzero/lib/benchmark.py \
    --bigquery_dataset_table_name="" \
    --data_downloads="gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/cifar-10-batches-bin" \
    --ml_framework_build_label=v2-nightly-gpu \
    --execution_label="${label}" \
    --platform_name=kokoro-gcp \
    --system_name=n1-standard-8-1xA100 \
    --output_gcs_url="" \
    --benchmark_class_type= \
    --scratch_gcs_url= \
    --root_data_dir=/data \
    --benchmark_num_trials=2 \
    --bigquery_project_name="" \
    --git_repos="https://github.com/tensorflow/models.git;benchmark" \
    --python_path=models \
    --benchmark_methods=${benchmark_methods} \
    --result_upload_methods="" \
    --gcloud_key_file_url="${PERFZERO_GCLOUD_KEY_FILE_URL}" \
    --tpu_parameters=
}

function setup_docker() {
  label=$1
  dockerfile=$2

  echo "`date` Setting up ${label} docker..."
  sudo python3 perfzero/lib/setup.py \
    --gce_nvme_raid= \
    --docker_tag="${label}" \
    --gcloud_key_file_url= \
    --tensorflow_pip_spec=${tf_spec} \
    --dockerfile_path=${dockerfile}
  echo "`date` Finished setting up ${label} docker."
}

baseline_docker="docker/Dockerfile_ubuntu_1804_tf_cuda_11"
experiment_docker="docker/Dockerfile_ubuntu_cuda11_8_0_0_180"

setup_docker "control/tensorflow" ${baseline_docker}
run_benchmark "control/tensorflow" "control-8-0-4-30"

setup_docker "experiment/tensorflow" ${experiment_docker}
run_benchmark "experiment/tensorflow" "experiment-8-0-0-180"
