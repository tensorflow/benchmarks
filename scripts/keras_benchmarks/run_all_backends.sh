#!/bin/bash

declare -a backend_types=('tensorflow' 'theano' 'cntk')

if [ "$1" = "cpu" ]; then
   echo "$1"
   for bt in "${backend_types[@]}";do
      python -c "from keras import backend"
      KERAS_BACKEND=$bt
      sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
      echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"
      python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND --cpu_num_cores='1'
      --cpu_memory='3.75' --cpu_memory_info='GB' --platform_type='GCP' --platform_machine_type='n1-standard-1'
    done
fi
