#!/bin/bash -e

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

MASTER_NAME="$1"
WORKER_PREFIX="$2"
WORKER_COUNT=2
ZONE=europe-west1-b
CONFIG="$3"

$GCLOUD config set project "tensorflow-testing"
$GCLOUD config set compute/zone "$ZONE"


WORKER_NAMES=$(for i in $(seq 1 $WORKER_COUNT); do echo $WORKER_PREFIX$i; done)

# Delete instances
$GCLOUD compute instances delete $MASTER_NAME --zone=$ZONE -q
$GCLOUD compute instances delete $WORKER_NAMES --zone=$ZONE -q

# Delete firewall rule
$GCLOUD compute firewall-rules delete "${MASTER_NAME}-access" -q

# Release external IPs
$GCLOUD compute addresses delete ${MASTER_NAME} --region=europe-west1 -q
$GCLOUD compute addresses delete ${WORKER_NAMES} --region=europe-west1 -q

# Delete GCS config
$GSUTIL rm $CONFIG
