# Ubuntu 18.04 Python3 with CUDA 10 and the following:
#  - Installs tf-nightly-gpu-2.0-preview
#  - Installs requirements.txt for tensorflow/models
#  - Install bazel for building TF from source

FROM nvidia/cuda:10.0-base-ubuntu18.04 as base
ARG tensorflow_pip_spec="tf-nightly-gpu-2.0-preview"
ARG extra_pip_specs=""
ARG local_tensorflow_pip_spec=""

COPY ${local_tensorflow_pip_spec} /${local_tensorflow_pip_spec}

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0  \
        libcudnn7-dev=7.6.2.24-1+cuda10.0  \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        software-properties-common \
        unzip \
        lsb-core \
        curl \
        && \
  find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN apt-get update && \
    apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
    libnvinfer-dev=5.1.5-1+cuda10.0 \
    && apt-get clean

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Add google-cloud-sdk to the source list
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install extras needed by most models
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
      ca-certificates \
      wget \
      htop \
      zip \
      google-cloud-sdk

# Install / update Python
# (bulding TF needs py2 even if building for Python3 as of 06-AUG-2019)
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv \
      python

# Upgrade pip, need to use pip3 and then pip after this or an error
# is thrown for no main found.
RUN pip3 install --upgrade pip
# setuptools upgraded to fix install requirements from model garden.
RUN pip install wheel
RUN pip install --upgrade setuptools google-api-python-client pyyaml google-cloud google-cloud-bigquery google-cloud-datastore mock
RUN pip install absl-py
RUN pip install --upgrade --force-reinstall ${tensorflow_pip_spec} ${extra_pip_specs}
RUN pip install tfds-nightly
RUN pip install -U scikit-learn

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 freeze

# Install bazel
ARG BAZEL_VERSION=0.24.1
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

RUN git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src
