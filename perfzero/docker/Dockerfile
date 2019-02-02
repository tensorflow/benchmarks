FROM tensorflow/tensorflow:nightly-gpu-py3

WORKDIR /root
ENV HOME /root

# Add google-cloud-sdk to the source list
RUN apt-get install -y curl
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update

# Install extras needed by most models
RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      htop \
      zip \
      google-cloud-sdk \
      vim \
      unzip

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
RUN pip3 install --upgrade setuptools google-api-python-client google-cloud google-cloud-bigquery
RUN pip3 install wheel absl-py
RUN pip3 install --upgrade --force-reinstall tf-nightly-gpu

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
