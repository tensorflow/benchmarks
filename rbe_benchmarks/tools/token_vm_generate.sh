#!/bin/bash

set -e
set -x

BUCKET=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/BUCKET -H "Metadata-Flavor: Google")
FILENAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/FILENAME -H "Metadata-Flavor: Google")


# Install requirements for Kubernetes
apt-get update && apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y kubelet kubeadm kubectl kubernetes-cni

# Generate Kubernetes cluster token and copy it to GCS.
if [ ! -f /etc/kubernetes/admin.conf ]; then
  kubeadm reset  # Reset kubeadm when running init first time
fi

mkdir -p /home/ubuntu/
kubeadm token generate > /home/ubuntu/"${FILENAME}"

gsutil cp /home/ubuntu/"${FILENAME}" gs://"${BUCKET}"
