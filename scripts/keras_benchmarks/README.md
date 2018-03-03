# Keras Benchmarks

## Goal
Create a simple user workflow to run Keras benchmarks in a platform agnostic manner. We want users
to be able to benchmark different keras models in a seamless manner.

Note: This work is still in progress.


## Instance configuration

For now we will start by running these benchmarks on GCE instances and uploading results to BiqQuery
tables. To run benchmarks against the models present in the models directory you need to bring up
GCE instances with the following configurations:
```
Machine type: n1-standard-16 (16 vCPUs, 60 GB memory)
CPU platform: Intel Sandy Bridge
GPUs 4 x NVIDIA Tesla K80 (if you are benchmarking models on GPU)
Allow full access to BigQuery APIs
Allow ssh connections along with HTTP, HTTPS traffic.
```
Most of the above configurations should be part of the create instance workflow on GCE.

## Benchmarking Keras models

Log into the GCE instance using SSH. This should be seamless if you use the GCP UI to open a SSH
window.
Tip: Use tmux or an equivalent session manager to maintain stateful sessions. Running the below
scripts can take time and you might be disconnected from the GCE instance in the middle of a run.

Clone the benchmarks repo to access the start up scripts for running models.
```
https://github.com/tensorflow/benchmarks.git
git checkout -b keras-benchmarks origin/keras-benchmarks
```

Depending on if you want to benchmark models against CPUs or GPUs you can run setup_cpu.sh or
setup_gpu.sh. If you run setup_gpu.sh ensure that the NVIDIA GPU drivers have been successfully
installed by running the following command:
```
nvidia-smi
```
You should see the available GPUs at this point.
Note: we should ideally have conda environments and execute these scripts in two separate
environments. However CNTK installation had issues with conda environments and I opted for a
manual install.

After a successful installation of the required packages you can run a script depending on the
backend you want to benchmark against. backend-type can be tensorflow, theano or cntk.
```
sh run_<backend-type>_backend.sh
```

The above script runs all the models in the models/ directory.

## Metrics

We are currently recording the total time taken to run N epochs. N here is chosen to be 3. We run
the model for 4 epochs but discard the first epoch since the time interval will include time taken
to build the model.

We upload the model's metadata along with the metrics generated to our table in BigQuery.
