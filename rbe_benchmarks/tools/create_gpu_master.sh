#!/bin/bash

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud

# Zone must have GPU quota.
IMAGE_PROJECT=ubuntu-os-cloud
IMAGE_FAMILY=ubuntu-1604-lts

# Directory where this script is located.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


# Start master node
$GCLOUD compute instances create $MASTER_NAME \
  --scopes=storage-full \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --zone=$ZONE \
  --machine-type=n1-standard-1 \
  --can-ip-forward \
  --address="${MASTER_IP}" \
  --tags=benchmark-cluster \
  --boot-disk-size=100GB \
  --metadata="TOKEN=${TOKEN},MASTER_IP=${MASTER_IP},MASTER_PORT=${MASTER_PORT},CONFIG_LOCATION=${CONFIG_LOCATION}" \
  --metadata-from-file "startup-script=${SCRIPT_DIR}/setup_gpu_master.sh"

echo "To connect to the Kubernetes cluster remotely, copy over the kubeconfig "
echo "file using"
echo ""
echo "gcloud compute copy-files ubuntu@$MASTER_NAME:~/.kube/config \\"
echo "~/custom_kube_config.conf --zone=$ZONE"
echo ""
echo "Then, you can set KUBECONFIG=~/custom_kube_config.conf to run kubectl"
echo "commands against the master created here."
