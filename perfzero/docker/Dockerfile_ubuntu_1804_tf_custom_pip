# Ubuntu 18.04 Python3.6 with CUDA 10 and the following:
#  - Installs custom TensorFlow pip package
#  - Installs requirements.txt for tensorflow/models

# NOTE: Branched from Dockerfile_ubuntu_1804_tf_v1 with changes relevant to
# tensorflow_pip_spec. When updating please keep the difference minimal.

FROM nvidia/cuda:10.0-base-ubuntu18.04 as base

# Location of custom TF pip package, must be relative to docker context.
# Note that the version tag in the name of wheel file is meaningless.
ARG tensorflow_pip_spec="resources/tensorflow-0.0.1-cp36-cp36m-linux_x86_64.whl"
ARG extra_pip_specs=""
ARG local_tensorflow_pip_spec=""

COPY ${local_tensorflow_pip_spec} /${local_tensorflow_pip_spec}

COPY ${tensorflow_pip_spec} /tensorflow-0.0.1-cp36-cp36m-linux_x86_64.whl

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        software-properties-common \
        unzip \
        lsb-core \
        curl

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0 \
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

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv


# Setup Python3 environment
RUN pip3 install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
RUN pip3 install wheel
RUN pip3 install --upgrade setuptools google-api-python-client pyyaml google-cloud google-cloud-bigquery
RUN pip3 install absl-py
RUN pip3 install --upgrade --force-reinstall /tensorflow-0.0.1-cp36-cp36m-linux_x86_64.whl ${extra_pip_specs}
RUN pip3 install tfds-nightly
RUN pip3 install -U scikit-learn

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 freeze
