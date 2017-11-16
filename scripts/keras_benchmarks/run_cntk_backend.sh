#!/bin/bash

# Keras CNTK Backend
python -c "from keras import backend"
KERAS_BACKEND=cntk
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

# Use "cpu_config", "gpu_config" and "multi_gpu_config" as command line arguments to load the right
# config file.
if [ "$1" = "multi_gpu_config" ]; then
  mpiexec -n 4 python benchmarks/scripts/keras_benchmarks/run_benchmark.py "$1"
fi

python benchmarks/scripts/keras_benchmarks/run_benchmark.py "$1"

