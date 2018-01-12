#!/bin/bash -e

GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

CLUSTER_PREFIX=$1
TOKEN_GCS_BUCKET=$2
CONFIG_GCS_BUCKET=$3
CLUSTER_NUMBER=$(cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 8 | head -n 1)
CLUSTER="$CLUSTER_PREFIX"-"$CLUSTER_NUMBER"
FILE_NAME="config-$CLUSTER_NUMBER"

$GSUTIL cp gs://$TOKEN_GCS_BUCKET/permanent-token $HOME/token
TOKEN=$(cat "$HOME"/token)

echo "New cluster name is $CLUSTER."
echo "cluster=$CLUSTER" > "$_SETUP_OUTPUT"
echo "New config file name is $FILE_NAME."
echo "config_location=gs://$CONFIG_GCS_BUCKET/${FILE_NAME}" >> "$_SETUP_OUTPUT"
echo "token=$TOKEN" >> "$_SETUP_OUTPUT"
touch "$_SETUP_DONE"
