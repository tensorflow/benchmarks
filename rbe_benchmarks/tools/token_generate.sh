#!/bin/bash -e

GCLOUD=/var/gcloud/google-cloud-sdk/bin/gcloud
GSUTIL=/var/gcloud/google-cloud-sdk/bin/gsutil

$GCLOUD config set project tensorflow-testing
$GCLOUD config set compute/zone europe-west1-b

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

VM_NAME="$1"
FILE_NAME="$2"
BUCKET="$3"
$GCLOUD compute instances create $VM_NAME --scopes=storage-full \
  --metadata="BUCKET=${BUCKET},FILENAME=${FILE_NAME}" \
  --metadata-from-file "startup-script=${SCRIPT_DIR}/token_vm_generate.sh"

set +e
while true; do
  if [[ "$n" -gt 5 ]]; then
    echo "Timed out after 5 minutes waiting."
    exit 1
  else
    $GSUTIL -q stat gs://${BUCKET}/${FILE_NAME}
    if [[ $? -eq 0 ]]; then
     echo "Token file exists!";
     break;
    fi
  fi
  echo "Token file is not pushed after $n minutes."
  ((n++))
  sleep 60
done
set -e

$GSUTIL cp gs://${BUCKET}/${FILE_NAME} $HOME/token


TOKEN=$(cat "$HOME"/token)

echo "token=$TOKEN" > "$_SETUP_OUTPUT"
touch "$_SETUP_DONE"
