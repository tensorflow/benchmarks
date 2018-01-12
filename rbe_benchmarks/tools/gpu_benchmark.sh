#!/bin/bash -e
CLUSTER="$1"
ZONE="$2"
PROJECT="$3"
DOCKER_IMAGE="$4"
CONFIG_FILE_BASE="$5"
CUDA_LIB_DIR="$6"
NVIDIA_LIB_DIR="$7"
BENCHMARK_NAME="$8"
K8S_CONFIG_LOCATION="$9"

/usr/local/bin/pip install --user pyyaml

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# TODO(jiyang): find a reliable way to get IP
ip=$(dig +short myip.opendns.com @resolver1.opendns.com)


# Parse firewall rule to add worker to access K8S GPU cluster if needed
$GCLOUD config set project "$PROJECT"
$GCLOUD config set compute/zone "$ZONE"

echo "$("${GCLOUD}" compute firewall-rules describe "${CLUSTER}"-access)" | \
  tr '\n' ' ' | \
  grep -o -P '(?<=sourceRanges: - ).*(?= targetTags:)' > /tmp/range
sed -i 's/ - /,/g' /tmp/range
ip_list=$(</tmp/range)

if [[ $ip_list == *"$ip"* ]]; then
  echo "The worker address is found in the firewall rule."
else
  /var/gcloud/google-cloud-sdk/bin/gcloud compute firewall-rules update \
    "$1"-access --source-ranges="$ip_list,$ip/32"
  echo "The worker address is added in the firewall rule."
fi

# Wait for 5 minutes or until the config files are pushed.
# Note that for "with cluster" case, this should pass in first try.
set +e
n=0
while true; do
  if [[ "$n" -gt 5 ]]; then
    echo "Timed out after 5 minutes waiting."
    exit 1
  else
    $GSUTIL -q stat $K8S_CONFIG_LOCATION
    if [[ $? -eq 0 ]]; then
     echo "Config File exists!";
     break;
    fi
  fi
  echo "File is not pushed after $n minutes."
  ((n++))
  sleep 60
done

# Download config file from gcs bucket
$GSUTIL cp "$K8S_CONFIG_LOCATION" /tmp/gpu-cluster.conf
export KUBECONFIG=/tmp/gpu-cluster.conf

set -e

# Run Benchmark Test
python rbe_benchmarks/tools/generate_yml.py \
  --docker_image="$DOCKER_IMAGE" \
  --benchmark_configs_file="$CONFIG_FILE_BASE" \
  --benchmark_config_output=/tmp/generated_gpu_benchmark.yml \
  --cuda_lib_dir="$CUDA_LIB_DIR" \
  --nvidia_lib_dir="$NVIDIA_LIB_DIR"

# The wait-cluster-to-be-ready logic is in the script.
python rbe_benchmarks/tools/run_distributed_benchmarks.py \
  --benchmark_name="$BENCHMARK_NAME" \
  --kubernetes_config_file=/tmp/generated_gpu_benchmark.yml
