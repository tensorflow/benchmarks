#!/bin/bash

# setup script for running benchmarks on GPU

# Install pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

# Install Nvidia drivers CUDA 8
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0 -y

# [Instructions from GCP docs:https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver]
# Use nvidia-smi to verify that the drivers have been installed

# set the CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64

#Install cudnn library
# TODO(anjalisridhar): the cudann library was downloaded to the local machine. try using curl
gsutil cp gs://keras-benchmarks/libcudnn6_6.0.21-1+cuda8.0_amd64.deb .
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb

# CUB for CNTK
wget https://github.com/NVlabs/cub/archive/1.4.1.zip
sudo apt-get install unzip -y
unzip ./1.4.1.zip
sudo cp -r cub-1.4.1 /usr/local

# CNTK requires cudnn installation to be in a specific directory
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar -xzvf ./cudnn-8.0-linux-x64-v6.0.tgz
sudo mkdir /usr/local/cudnn-6.0
sudo cp -r cuda /usr/local/cudnn-6.0
export LD_LIBRARY_PATH=/usr/local/cudnn-6.0/cuda/lib64:$LD_LIBRARY_PATH

# MPI installation
sudo apt-get install openmpi-bin -y

# Install CNTK GPU version
pip install https://cntk.ai/PythonWheel/GPU/cntk-2.2-cp27-cp27mu-linux_x86_64.whl

# Install other pacakges required for CNTK
sudo apt-get install libopencv-dev python-opencv -y

# Install keras
sudo pip install keras

# Install required pacakges for TF-GPU
sudo apt-get install python-dev python-pip libcupti-dev

# Install tensorflow GPU version
sudo pip install tensorflow-gpu

sudo pip install git+git://github.com/fchollet/keras.git --upgrade

# Install google-cloud tools
echo "Installing Google Cloud tools"
sudo pip install google-cloud
sudo pip install google-cloud-bigquery

# Install h5py
sudo pip install h5py
