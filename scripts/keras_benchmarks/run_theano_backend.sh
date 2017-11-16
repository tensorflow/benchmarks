#!/bin/bash

# Keras Theano Backend
python -c "from keras import backend"
KERAS_BACKEND=cntk
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

# Use "cpu_config", "gpu_config" and "multi_gpu_config" as command line arguments to load the right
# config file.
if [ "$1" = "cpu_config" ]; then
  python benchmarks/scripts/keras_benchmarks/run_benchmark.py "$1"
else
  echo "GPU mode for Theano backend is not supported currently by the keras benchmarks script."
fi