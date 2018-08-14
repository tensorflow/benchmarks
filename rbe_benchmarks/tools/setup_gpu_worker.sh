#!/bin/bash
# Startup script to run on Google Cloud instances.
# This script is used by create_gpu_workers.sh.
set -e
set -x

TOKEN=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/TOKEN -H "Metadata-Flavor: Google")
MASTER_IP=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/MASTER_IP -H "Metadata-Flavor: Google")
MASTER_PORT=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/MASTER_PORT -H "Metadata-Flavor: Google")

# Install requirements for Kubernetes
apt-get update && apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y docker-engine
apt-get install -y kubelet kubeadm kubectl kubernetes-cni


# Install cuda
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
requires_reboot=false
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
  requires_reboot=true
fi
apt-get install -y nvidia-cuda-toolkit

# Join Kubernetes cluster.
if [ "$requires_reboot" == true ]; then
  kubeadm reset  # Reset only once, before restarting
fi
kubeadm join --token "${TOKEN}" "${MASTER_IP}:${MASTER_PORT}"

# Update systemd file for kubelet to add gpu feature gate.
SYSTEMD_FILE='/etc/systemd/system/kubelet.service.d/10-kubeadm.conf'
GPU_FEATURE="--feature-gates=Accelerators=true"
if ! grep -q "${GPU_FEATURE}" $SYSTEMD_FILE; then
  sed -i "/ExecStart=\/usr\/bin\/kubelet/ s/$/ ${GPU_FEATURE}/" "${SYSTEMD_FILE}"
  requires_reboot=true
fi

# Reboot after cuda installation and to pick up systemd file changes.
if [ "$requires_reboot" == true ]; then
  reboot
fi
