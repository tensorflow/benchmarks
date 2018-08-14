#!/bin/bash -e

/var/gcloud/google-cloud-sdk/bin/gcloud container clusters get-credentials \
  "$1" --zone "$2" --project "$3"

/var/gcloud/google-cloud-sdk/bin/gcloud container clusters delete "$1" \
  --zone "$2" --project "$3" -q
