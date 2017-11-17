#!/bin/bash

# setup script for running benchmarks on CPU

sudo apt-get update

# Install pip package manager
echo "Installing pip"
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

sudo apt-get install bzip2

# Install conda environment manager
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod 777 miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
conda create -q -n benchmarks-environment python="2.7" numpy scipy
source activate benchmarks-environment
# set library path
export LD_LIBRARY_PATH=$HOME/miniconda/envs/test-environmcondent/lib/:$LD_LIBRARY_PATH

# Install Pillow package
conda install pil

# Install Theano
echo "Installing Theano"
pip install theano

# Install MKL library for Theano
conda install mkl-service

# Install g++
sudo apt-get install g++ -y

# Install tensorflow
echo "Installing Tensorflow"
pip install tensorflow

# Install CNTK
echo "Installing CNTK"
pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.2-cp27-cp27mu-linux_x86_64.whl

# Install OpenCV
sudo apt-get install libopencv-dev python-opencv -y

# Install open mpi
rm -rf ~/mpi
mkdir ~/mpi
pushd ~/mpi
wget http://cntk.ai/PythonWheel/ForKeras/depends/openmpi_1.10-3.zip
sudo apt-get install unzip -y
unzip ./openmpi_1.10-3.zip
sudo dpkg -i openmpi_1.10-3.deb
popd

# Install Keras
echo "Installing Keras"
pip install keras

# Install git
echo "Installing Git"
sudo apt-get install git -y

# Install google-cloud tools
echo "Installing Google Cloud tools"
pip install google-cloud
pip install google-cloud-bigquery

# Install h5py
pip install h5py
