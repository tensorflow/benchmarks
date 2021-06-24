#!/bin/bash
set -e
set -x

# To run this script from a GCP VM / host connected to GPUs:
# git clone https://github.com/tensorflow/benchmarks.git
# cd benchmarks
# bash perfzero/dockertest/resnet50_synth.sh
# Output log files/results will be stored at perfzero/workspace/output/

# Modify INPUT_PARAMS variables below to tweak the tf whl under test / benchmark methods / dataset paths.
# You can comment out "build_docker" call at the end, if the docker's already built.

## INPUT PARAMS: start

# Acceptable formats for TF_PIP_SPEC
# pypi nightlies: tf-nightly-gpu==2.6.0.dev20210521
# gcs path to whls: gs://some-path-to-tf.whl
# Local path to whl: file://some-local-path-to-tf.whl
TF_PIP_SPEC="tf-nightly-gpu==2.6.0.dev20210624"

# Path to GCS or local files containing the input datasets (if they need to be fetched into the docker).
DATA_DOWNLOADS=""

# Comma separated list of strings.
BENCHMARK_METHODS="official.benchmark.keras_imagenet_benchmark.Resnet50KerasBenchmarkSynth.benchmark_1_gpu_fp16"

# If either the tf_pip_spec or data downloads reference private GCP, then we
# need to set GCLOUD_KEY_FILE_URL to a credentials file.
GCLOUD_KEY_FILE_URL=""

# Commit id under repository tensorflow/models, branch='benchmark' which has the benchmarks.
MODELS_GIT_HASH="169e4051aef247c27a95748a8015b2f35f509e1a"
## INPUT PARAMS: end



build_docker() {
        echo "building docker"
sudo python3 perfzero/lib/setup.py \
        --dockerfile_path=docker/Dockerfile_ubuntu_1804_tf_cuda_11 \
        --tensorflow_pip_spec="${TF_PIP_SPEC}" \
        --gcloud_key_file_url="${GCLOUD_KEY_FILE_URL}" \
        --extra_docker_build_args=
sudo docker images
}


run_benchmark() {
        echo "running benchmark"
        benchmark_tag=$1
        env_var=$2
        
sudo nvidia-docker run \
        -v ${PWD}:/workspace \
        -v /data:/data \
        -e PERFZERO_EXECUTION_MODE=test \
        -e TF_ENABLE_LEGACY_FILESYSTEM=1 \
        -e ${env_var} \
        perfzero/tensorflow python3 \
        /workspace/perfzero/lib/benchmark.py \
        --root_data_dir=/data \
        --bigquery_dataset_table_name="" \
        --benchmark_class_type= \
        --ml_framework_build_label=v2-nightly-gpu-${benchmark_tag} \
        --execution_label=test-benchmark \
        --platform_name=kokoro-gcp \
        --system_name=n1-standard-8-1xV100 \
        --output_gcs_url="" \
        --benchmark_num_trials=1 \
        --scratch_gcs_url= \
        --bigquery_project_name="" \
        --git_repos="https://github.com/tensorflow/models.git;benchmark;${MODELS_GIT_HASH}" \
        --data_downloads="${DATA_DOWNLOADS}"\
        --python_path=models \
        --benchmark_methods="${BENCHMARK_METHODS}" \
        --gcloud_key_file_url="${GCLOUD_KEY_FILE_URL}"
}

build_docker
run_benchmark "control" "TF_CUDNN_USE_FRONTEND=false"
run_benchmark "experiment" "TF_CUDNN_USE_FRONTEND=true"
