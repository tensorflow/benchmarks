#!/bin/bash

# Script to start Compute Engine VM Instances with GPUs and set up
# Kubernetes on them. This script just starts Kubernetes worker nodes.
# Kubernetes master node needs to be started separately.
#
# To run, first set environment variables TOKEN, MASTER_IP, MASTER_PORT,
# WORKER_PREFIX, and GPUS_PER_WORKER.
# To get a new token, run
#
# kubeadm token create
#
# To get a list of existing tokens on the master, run:
#
# kubeadm token list


# How many instances to create. Instances will be named as
# benchmark-cluster-node-1, benchmark-cluster-node-2, etc..
WORKER_NAMES=$(for i in $(seq 1 $WORKER_COUNT); do echo $WORKER_PREFIX$i; done)

ZONE="${ZONE:europe-west1-b}"
IMAGE_PROJECT=ubuntu-os-cloud
IMAGE_FAMILY=ubuntu-1604-lts

# Directory where this script is located.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
# Start worker nodes
$GCLOUD beta compute instances create $WORKER_NAMES \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --can-ip-forward \
  --accelerator=type=nvidia-tesla-k80,count="${GPUS_PER_WORKER}" \
  --maintenance-policy=TERMINATE \
  --tags=benchmark-cluster \
  --boot-disk-size=200GB \
  --scopes=https://www.googleapis.com/auth/datastore,https://www.googleapis.com/auth/devstorage.read_only \
  --metadata="TOKEN=${TOKEN},MASTER_IP=${MASTER_IP},MASTER_PORT=${MASTER_PORT}" \
  --metadata-from-file "startup-script=${SCRIPT_DIR}/setup_gpu_worker.sh"
