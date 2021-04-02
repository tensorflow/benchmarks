# Ubuntu 18.04 Python3 with CUDA 11 and the following:
#  - Installs tf-nightly-gpu (this is TF 2.3)
#  - Installs requirements.txt for tensorflow/models

FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 as base
ARG tensorflow_pip_spec="tf-nightly-gpu"
ARG local_tensorflow_pip_spec=""
ARG extra_pip_specs=""

# setup.py passes the base path of local .whl file is chosen for the docker image.
# Otherwise passes an empty existing file from the context.
COPY ${local_tensorflow_pip_spec} /${local_tensorflow_pip_spec}

# Pick up some TF dependencies
# cublas-dev and libcudnn7-dev only needed because of libnvinfer-dev which may not
# really be needed.
# In the future, add the following lines in a shell script running on the
# benchmark vm to get the available dependent versions when updating cuda
# version (e.g to 10.2 or something later):
# sudo apt-cache search cuda-command-line-tool
# sudo apt-cache search cuda-cublas
# sudo apt-cache search cuda-cufft
# sudo apt-cache search cuda-curand
# sudo apt-cache search cuda-cusolver
# sudo apt-cache search cuda-cusparse

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-tools-11-0 \
        cuda-toolkit-11-0 \
        libcudnn8=8.0.4.30-1+cuda11.0  \
        libcudnn8-dev=8.0.4.30-1+cuda11.0  \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        software-properties-common \
        unzip \
        lsb-core \
        curl

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Add google-cloud-sdk to the source list
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install extras needed by most models
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      wget \
      htop \
      zip \
      google-cloud-sdk

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv

# Upgrade pip, need to use pip3 and then pip after this or an error
# is thrown for no main found.
RUN pip3 install --upgrade pip
# setuptools upgraded to fix install requirements from model garden.
RUN pip install wheel
RUN pip install --upgrade setuptools google-api-python-client==1.8.0 pyyaml google-cloud google-cloud-bigquery google-cloud-datastore mock
RUN pip install absl-py
RUN pip install --upgrade --force-reinstall ${tensorflow_pip_spec} ${extra_pip_specs}
RUN pip install tfds-nightly
RUN pip install -U scikit-learn
# Install dependnecies needed for tf.distribute test utils
RUN pip install dill tblib portpicker

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install tf-estimator-nightly
RUN pip install tensorflow-text-nightly

# RUN nvidia-smi

RUN nvcc --version


RUN pip freeze
