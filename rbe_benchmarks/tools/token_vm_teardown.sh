#!/bin/bash -e

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

$GCLOUD config set project tensorflow-testing
$GCLOUD config set compute/zone europe-west1-b

VM_NAME="$1"
FILE_NAME="$2"
BUCKET="$3"

echo "Delete $VM_NAME."
$GCLOUD compute instances delete $VM_NAME -q

# Uncomment below to clear the generated token.
# echo "Delete gs://${BUCKET}/${FILE_NAME}."
# $GSUTIL rm gs://${BUCKET}/${FILE_NAME}
