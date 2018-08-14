#!/bin/bash -e

GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

CLUSTER_PREFIX=$1
CLUSTER_NUMBER=$(cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 8 | head -n 1)
CLUSTER="$CLUSTER_PREFIX"-"$CLUSTER_NUMBER"

echo "New cluster name is $CLUSTER."
echo "cluster=$CLUSTER" > "$_SETUP_OUTPUT"
touch "$_SETUP_DONE"
