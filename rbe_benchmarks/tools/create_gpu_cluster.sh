#!/bin/bash
set -e
set -x

me="$(basename "$0")"
usage="usage: $me [-m master_name] [-w worker_prefix] [-c worker_count] \\
                  [-g gpus_per_worker] [-t token]

Creates a Kubernetes cluster on Google Compute Engine. Specifically, this script
will start Kubernetes master and Kubernetes workers.

Note: To run kubectl commands that connect to the newly started cluster, you
would need to copy over cluster config. After master starts up, copy over the
config file using

gcloud compute copy-files ubuntu@<master_name>:~/.kube/config \\
    ~/custom_kube_config.conf --zone=europe-west1-b

For example:
gcloud compute scp ubuntu@benchmark-cluster-master:/home/ubuntu/.kube/config \\
    ~/custom_kube_config.conf --zone=europe-west1-b

Then, you can set KUBECONFIG=~/custom_kube_config.conf to run kubectl commands
against the master created here.

-m master_name
  Specifies name of Kubernetes master VM Instance to create.

-w worker_prefix
  Specifies prefix of Kubernetes worker VM Instances to create. Kubernetes VM
  Instances will be named worker_prefix0, worker_prefix1, ..., worker_prefixN-1
  where N is the worker_count.

-c worker_count
  Number of Kubernetes workers to create.

-g gpus_per_worker
  How many GPUs to request for each Kubernetes worker VM Instance.
  Must be 1, 2, 4, or 8.

-i master_ip
  gcloud External IP address to use for Kubernetes master.

-t token
  kubernetes token to create cluster with.

-f config_location
  Config location on GCS bucket.

Examples:

# Create Kubernetes cluster with 10 workers, 2 GPUs each:
$me -c 10 -g 2
"

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

$GCLOUD config set project tensorflow-testing
$GCLOUD config set compute/zone europe-west1-b

master_name=benchmark-cluster-master
worker_prefix=benchmark-cluster-worker-
worker_count=2
gpus_per_worker=8

# Zone must have GPU quota.
export ZONE=europe-west1-b
export REGION=europe-west1

while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -m|--master_name)
      master_name="$2"
      shift
      ;;
    -w|--worker_prefix)
      worker_prefix="$2"
      shift
      ;;
    -c|--worker_count)
      worker_count="$2"
      shift
      ;;
    -g|--gpus_per_worker)
      gpus_per_worker="$2"
      ;;
    -i|--master_ip)
      master_ip="$2"
      ;;
    -t|--token)
      given_token="$2"
      ;;
    -f|--config_location)
      config_location="$2"
      ;;
    -*)
      echo "Unrecognized option $1"
      echo "$usage"
      exit 1
      ;;
    *)
      ;;
  esac
  shift
done

# Create external ip address.
if [[ -z "$master_ip" ]]; then
  echo "No master ip passed in, creating or reusing one with name ${master_name}"
  # If address does not exist for ${master_name}, then the command below will
  # exit with error status. We don't want to stop entire script at that point.
  # Instead, we create a new address.
  set +e
  ip_status="$($GCLOUD compute addresses describe ${master_name} --region ${REGION} --format='get(status)')"
  set -e
  if [[ "$ip_status" == "IN_USE" ]]; then
    echo "Failed to create a cluster: external IP with name ${master_name} already exists and is in use."
    exit 1
  elif [[ -z "$ip_status" ]]; then
    master_ip="$($GCLOUD compute addresses create ${master_name} --region ${REGION} --format='get(address)')"
    echo "Created a new ip for master: ${master_ip}"
  else
    master_ip="$($GCLOUD compute addresses describe ${master_name} --region ${REGION} --format='get(address)')"
    echo "Reusing existing ip for master: ${master_ip}"
  fi
fi

# Set variables to be used by create_gpu_master.sh and
# create_gpu_workers.sh.
export MASTER_NAME="${master_name}"
export WORKER_PREFIX="${worker_prefix}"
export WORKER_COUNT="${worker_count}"
export GPUS_PER_WORKER="${gpus_per_worker}"
export MASTER_IP="${master_ip}"
export MASTER_PORT=6443
export CONFIG_LOCATION="${config_location}"

# Create a token for the Kubernetes cluster
if [[ -z "$given_token" ]]; then
  export TOKEN="$(kubeadm token generate)"
else
  export TOKEN="$given_token"
fi

# Directory where this script is located.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

${SCRIPT_DIR}/create_gpu_master.sh
${SCRIPT_DIR}/create_gpu_workers.sh

# Add a firewall rule to allow workers to communicate with the master.
# First, get worker IPs.
worker_ips=$($GCLOUD compute instances list --regexp ${worker_prefix}.* \
  --format="get(networkInterfaces.accessConfigs[0].natIP)" \
  --zones=europe-west1-b)
worker_names=$($GCLOUD compute instances list --regexp ${worker_prefix}.* \
  --format="get(name)" --zones=europe-west1-b)

# Create static IP addresses for workers.
# TODO(annarev): reuse static ips if they already exist.
comma_separated_worker_ips=$(echo "${worker_ips}" | awk '{print $1}' | paste -s -d, - )
echo "Reserving static ips for workers: ${worker_ips}"
$GCLOUD compute addresses create ${worker_names} \
    --addresses "${comma_separated_worker_ips}" \
    --region "${REGION}"

# Add /32 suffix to each IP and separate IPs with commas
worker_ip_ranges=$(echo "${worker_ips}" | awk '{print $1 "/32"}' | paste -s -d, - )
ip_ranges="${master_ip}/32,${worker_ip_ranges}"
firewall_rule_name="${master_name}-access"
echo "Creating a firewall rule ${firewall_rule_name} with source ip ranges ${ip_ranges}"
$GCLOUD compute firewall-rules create "${firewall_rule_name}" \
    --allow tcp:6443 \
    --source-ranges="${ip_ranges}"\
    --target-tags=benchmark-cluster

# ASCIT specific data
echo "cluster_name=$master_name" > "$_SETUP_OUTPUT"
echo "config_location=$config_location" >> "$_SETUP_OUTPUT"
touch "$_SETUP_DONE"
