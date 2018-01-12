#!/bin/bash -e

/var/gcloud/google-cloud-sdk/bin/gcloud container clusters create "$1" \
  --image-type=UBUNTU \
  --zone="$2" \
  --project="$3" \
  --machine-type=n1-standard-8 \
  --tags=benchmark-cluster \
  --disk-size=200 \
  --num-nodes=2 \
  --scopes="https://www.googleapis.com/auth/datastore,https://www.googleapis.com/auth/devstorage.read_only"

echo "cluster_name=$1" > "$_SETUP_OUTPUT"

touch "$_SETUP_DONE"
