#!/bin/bash -e
/usr/local/bin/pip install --user pyyaml

/var/gcloud/google-cloud-sdk/bin/gcloud container clusters get-credentials \
  "$1" --zone "$2" --project "$3"

python rbe_benchmarks/tools/generate_yml.py \
  --docker_image="$4" \
  --benchmark_configs_file="$5" \
  --benchmark_config_output="$6"

python rbe_benchmarks/tools/run_distributed_benchmarks.py \
  --benchmark_name="$7" \
  --kubernetes_config_file="$6"
