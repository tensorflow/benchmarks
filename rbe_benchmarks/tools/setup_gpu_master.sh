#!/bin/bash
# Startup script to run on Google Cloud instances and used by
# create_gpu_master.sh.
# This script sets up Kubernetes master node.
set -e
set -x

TOKEN=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/TOKEN -H "Metadata-Flavor: Google")
MASTER_IP=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/MASTER_IP -H "Metadata-Flavor: Google")
MASTER_PORT=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/MASTER_PORT -H "Metadata-Flavor: Google")
CONFIG_LOCATION=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/CONFIG_LOCATION -H "Metadata-Flavor: Google")

# Install requirements for Kubernetes
apt-get update && apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y docker-engine
apt-get install -y kubelet kubeadm kubectl kubernetes-cni

# Setup the Kubernetes master
if [ ! -f /etc/kubernetes/admin.conf ]; then
  kubeadm reset  # Reset kubeadm when running init first time
fi
kubeadm init --apiserver-advertise-address "${MASTER_IP}" --token "${TOKEN}"

UBUNTU_HOME='/home/ubuntu'
mkdir $UBUNTU_HOME/.kube
sudo cp /etc/kubernetes/admin.conf $UBUNTU_HOME/.kube/config
sudo chown ubuntu:$(id -g) $UBUNTU_HOME/.kube/config
export KUBECONFIG=/home/ubuntu/.kube/config
kubectl apply -f https://git.io/weave-kube-1.6

gsutil cp /home/ubuntu/.kube/config "$CONFIG_LOCATION"
