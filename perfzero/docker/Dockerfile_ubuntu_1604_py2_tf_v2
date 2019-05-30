FROM tensorflow/tensorflow:nightly-gpu

WORKDIR /root
ENV HOME /root
ARG tensorflow_pip_spec="tf-nightly-gpu-2.0-preview"

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

RUN pip install --upgrade --force-reinstall ${tensorflow_pip_spec}

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN pip freeze
