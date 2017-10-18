#!/bin/bash

sudo apt-get install bzip2

wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod 777 miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda

# Useful for debugging any issues with conda
conda info -a

conda create -q -n benchmarks-environment python="2.7" numpy scipy

source activate benchmarks-environment

# set library path
export LD_LIBRARY_PATH=$HOME/miniconda/envs/test-environment/lib/:$LD_LIBRARY_PATH

# Install pip package manager
echo "Installing pip"
sudo python get-pip.py
l
# Install Pillow package
conda install pil

# Install Theano
echo "Installing Theano"
pip install theano

# Install tensorflow
echo "Installing Tensorflow"
pip install tensorflow
#sudo pip install tensorflow-gpu

# Install CNTK
echo "Installing CNTK"
pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.2-cp27-cp27mu-linux_x86_64.whl
            https://cntk.ai/PythonWheel/CPU-Only/cntk-2.2-cp27-cp27m-win_amd64.whl

#install open mpi
rm -rf ~/mpi
mkdir ~/mpi
pushd ~/mpi
wget http://cntk.ai/PythonWheel/ForKeras/depends/openmpi_1.10-3.zip
sudo apt-get install unzip
unzip ./openmpi_1.10-3.zip
sudo dpkg -i openmpi_1.10-3.deb
popd

# Install Keras
echo "Installing Keras"
pip install keras

# Install git
echo "Installing Git"
sudo apt-get install git

# Install google-cloud tools
echo "Installing Google Cloud tools"
pip install google-cloud
pip install google-cloud-bigquery

# Clone benchmarks repo
rm -rf benchmarks
git clone -b keras-benchmarks https://github.com/tensorflow/benchmarks.git

KERAS_BACKEND=tensorflow
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

python benchmarks/scripts/keras_benchmarks/run_benchmarks.py --keras_backend=$KERAS_BACKED

