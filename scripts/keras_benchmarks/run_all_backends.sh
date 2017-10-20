#!/bin/bash

python -c "from keras import backend"
KERAS_BACKEND=tensorflow
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
# TODO(anjalisridhar): look into adding this as a command line arg
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND

python -c "from keras import backend"
KERAS_BACKEND=theano
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
# TODO(anjalisridhar): look into adding this as a command line arg
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND

python -c "from keras import backend"
KERAS_BACKEND=cntk
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
# TODO(anjalisridhar): look into adding this as a command line arg
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND
