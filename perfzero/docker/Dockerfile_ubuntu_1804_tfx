# Ubuntu 18.04 Python3 with CUDA 10 and the following:
#  - Installs tf-nightly-gpu (this is TF 2.0)
#  - Installs requirements.txt for tensorflow/models
#
# NOTE: Branched from Dockerfile_ubuntu_1804_tf_v2 with changes for
# TFX benchmarks.

FROM nvidia/cuda:10.0-base-ubuntu18.04 as base
ARG tensorflow_pip_spec="tf-nightly-gpu"
ARG local_tensorflow_pip_spec=""
ARG extra_pip_specs=""

# Specifies the default package version to use if no corresponding commit_id
# override is specified.
# If "head", uses the GitHub HEAD version.
# If "release", uses the latest released version from PyPI, REGARDLESS of
# package-compatibility requirements (e.g. even if tfx requires
# tensorflow-model-analysis<0.22, if tensorflow-model-analysis==0.22.0 is
# the latest released version on PyPI, we will install that).
# Packages include: tfx, tensorflow-transform, tensorflow-model-analysis,
# tensorflow-data-validation, tensorflow-metadata, tfx-bsl
ARG default_package_version="head"

# Specifies the package version to use for the corresponding packages.
# If empty, uses the default specified by default_package_version.
# If "head", uses the GitHub HEAD version.
# If "release", uses the latest released version from PyPI, REGARDLESS of
# package-compatibility requirements.
# If "github_commit:<commit id>", uses the given commit ID from GitHub.
# If "github_tag:<tag>" uses the given tag from GitHub.
# If "pypi:<version string>", uses the given package version from PyPI.
ARG tfx_package_version=""
ARG tensorflow_transform_package_version=""
ARG tensorflow_model_analysis_package_version=""
ARG tensorflow_data_validation_package_version=""
ARG tensorflow_metadata_package_version=""
ARG tfx_bsl_version=""

# setup.py passes the base path of local .whl file is chosen for the docker image.
# Otherwise passes an empty existing file from the context.
COPY ${local_tensorflow_pip_spec} /${local_tensorflow_pip_spec}

# Pick up some TF dependencies
# cublas-dev and libcudnn7-dev only needed because of libnvinfer-dev which may not
# really be needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
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
        curl

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
RUN pip install --upgrade setuptools google-api-python-client pyyaml google-cloud google-cloud-bigquery google-cloud-datastore mock
RUN pip install absl-py
RUN if [ ! -z "${extra_pip_specs}" ]; then pip install --upgrade --force-reinstall ${extra_pip_specs}; fi
RUN pip install tfds-nightly
RUN pip install -U scikit-learn

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install yolk3k, for getting package versions from PyPI (so we can pull
# TFX from GitHub even when we need to install from the released version)
RUN pip install yolk3k

# Install protoc
RUN PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip; \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP; \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc; \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*'; \
    rm -f $PROTOC_ZIP;

# Install Bazel
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt update
RUN apt install bazel
# Create symlink to "python3" binary under the name "python" so Bazel doesn't complain about "python" not being found
RUN ln -s $(which python3) /usr/bin/python

SHELL ["/bin/bash", "-c"]
RUN \
  function install_package { \
    # e.g. "head" or "release" \
    default_version="$1"; \
    # e.g "tensorflow-model-analysis" \
    package_name="$2"; \
    # e.g "model-analysis" \
    package_repo_name="$3"; \
    # How this package should be installed if pulled from GitHub. \
    # "none" for no extra installation steps required \
    # "bdist_wheel" for python setup.py bdist_wheel \
    package_install_type=$4; \
    # e.g. "head" or "release" or "pypi:0.21.4" or "github:[commit hash]" \
    package_version="$5"; \
    \
    # e.g. "tensorflow_model_analysis" \
    package_name_underscores=${package_name//-/_}; \
    if [ "$package_version" == "" ]; then \
      package_version="$default_version"; \
    fi; \
    \
    commit_id=""; \
    pypi_version=""; \
    if [ "$package_version" == "head" ]; then \
      commit_id=$(git ls-remote https://github.com/tensorflow/${package_repo_name} refs/heads/master | cut -f1); \
      echo ${package_name}: latest commit from GitHub: ${commit_id}; \
    elif [ "$package_version" == "release" ]; then \
      pypi_version=$(yolk -V $package_name | head -n 1 | cut -d' ' -f 2-); \
      echo ${package_name}: latest version from PyPI: ${pypi_version}; \
    elif [ "${package_version:0:5}" == "pypi:" ]; then \
      pypi_version="${package_version:5}"; \
      echo ${package_name}: using specified PyPI version: ${pypi_version}; \
    elif [ "${package_version:0:7}" == "github:" ]; then \
      commit_id="${package_version:7}"; \
      echo ${package_name}: using specified GitHub commit: ${commit_id}; \
    else \
      echo Unknown package version for ${package_name}: ${package_version}; \
      exit 1; \
    fi; \
    \
    if [ "$commit_id" != "" ]; then \
      if [ "$package_install_type" == "none" ]; then \
        # Package doesn't need extra installation steps - install directly from GitHub. \
        pip install -e git+https://github.com/tensorflow/${package_repo_name}.git@${commit_id}#egg=${package_name_underscores}; \
        install_commands+=("pip install --force --no-deps -e git+https://github.com/tensorflow/${package_repo_name}.git@${commit_id}#egg=${package_name_underscores}"); \
        echo Installed ${package_name} from GitHub commit ${commit_id}; \
      elif [ "$package_install_type" == "bdist_wheel" ]; then \
        # Package needs extra installation steps. Clone from GitHub, then build and install. \
        git clone https://github.com/tensorflow/${package_repo_name}.git; \
        pushd ${package_repo_name}; \
        git checkout ${commit_id}; \
        if [ "$package_name" == "tfx" ]; then \
          echo Building TFX pip package from source; \
          sed -i 's@packages=packages,@packages=packages, package_data={package_name: ["benchmarks/datasets/chicago_taxi/data/taxi_1M.tfrecords.gz"]},@' setup.py; \
          package_build/initialize.sh; \
          python package_build/ml-pipelines-sdk/setup.py bdist_wheel; \
          python package_build/tfx/setup.py bdist_wheel; \
        else \
          echo Using python setup.py bdist_wheel to build package; \
          python setup.py bdist_wheel; \
        fi; \
        pip install dist/*.whl; \
        install_commands+=("pip install --force --no-deps ${PWD}/dist/*.whl"); \
        popd; \
        echo Built and installed ${package_name} from GitHub commit ${commit_id}; \	
      fi; \
      # Write GIT_COMMIT_ID attribute to the installed package. \
      package_path=$(python3 -c "import ${package_name_underscores}; print(list(${package_name_underscores}.__path__)[0])"); \
      echo "GIT_COMMIT_ID='${commit_id}'" >> ${package_path}/__init__.py; \
      install_commands+=("echo \"GIT_COMMIT_ID='${commit_id}'\" >> ${package_path}/__init__.py;"); \
    elif [ "$pypi_version" != "" ]; then \
      if [ "$package_name" == "tfx" ]; then \
        # Special handling for TFX - we want to install from GitHub, and get \
        # the data files as well (they are not included in the pip package). \
        # Install from the corresponding tag in GitHub. \
        echo Special handling for tfx: will install tfx from GitHub tag for version ${pypi_version}; \
        git clone --depth 1 --branch v${pypi_version} https://github.com/tensorflow/tfx.git; \
        pushd tfx; \
        echo Building TFX pip package from source; \
        sed -i 's@packages=packages,@packages=packages, package_data={package_name: ["benchmarks/datasets/chicago_taxi/data/taxi_1M.tfrecords.gz"]},@' setup.py; \
        package_build/initialize.sh; \
        python package_build/ml-pipelines-sdk/setup.py bdist_wheel; \
        python package_build/tfx/setup.py bdist_wheel; \
        pip install dist/*.whl; \
        install_commands+=("pip install --force --no-deps ${PWD}/dist/*.whl"); \
        popd; \
        echo Installed tfx from GitHub tag for version ${pypi_version}; \
      else \
        pip install ${package_name}==${pypi_version}; \
        install_commands+=("pip install --force --no-deps ${package_name}==${pypi_version}"); \
        echo Installed ${package_name} from PyPI version ${pypi_version}; \
      fi; \
    else \
      echo Neither commit_id nor pypi_version was set for ${package_name}; \
      exit 1; \
    fi; \
  }; \
  \
  # Array of commands to run post-installation. This is for forcing \
  # installation of packages without regard to the requirements of other \
  # packages. \
  # The first round of installations installs the packages and their \
  # requirements. This may result in some packages being re-installed at \
  # versions other than the requested versions due to requirements from \
  # other packages. \
  # The second round of installations via install_commands \
  # forces installations of the packages at the desired versions, ignoring \
  # any dependencies of these packages or other packages. Note that if there \
  # are incompatible package dependencies (e.g. tfx depends on \
  # apache-beam==2.21 and tensorflow-transform depends on apache-beam==2.22 \
  # then either could be installed depending on the installation order). \
  install_commands=(); \
  install_package "${default_package_version}" "tfx" "tfx" "bdist_wheel" "${tfx_package_version}"; \
  install_package "${default_package_version}" "tensorflow-transform" "transform" "none" "${tensorflow_transform_package_version}"; \
  install_package "${default_package_version}" "tensorflow-model-analysis" "model-analysis" "none" "${tensorflow_model_analysis_package_version}"; \
  install_package "${default_package_version}" "tensorflow-data-validation" "data-validation" "bdist_wheel" "${tensorflow_data_validation_package_version}"; \
  install_package "${default_package_version}" "tensorflow-metadata" "metadata" "bdist_wheel" "${tensorflow_metadata_package_version}"; \
  install_package "${default_package_version}" "tfx-bsl" "tfx-bsl" "bdist_wheel" "${tfx_bsl_package_version}"; \
  for cmd in "${install_commands[@]}"; do \
    echo Running "${cmd}"; \
    eval $cmd; \
  done;

# Uninstall the TensorFlow version that TFX / the TFX components installed, and
# force install the version requested.
RUN pip uninstall -y tensorflow
RUN pip install --upgrade --force-reinstall ${tensorflow_pip_spec}

RUN pip freeze
