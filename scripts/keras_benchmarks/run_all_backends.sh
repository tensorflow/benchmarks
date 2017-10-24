#!/bin/bash

declare -a backend_types=('tensorflow' 'theano' 'cntk')
TODO(anjalisridhar): Can we represent cpu and gpu info  in a more structured format.ma
for bt in "${backend_types[@]}";do
      python -c "from keras import backend"
      KERAS_BACKEND=$bt
      sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
      echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"
      if [ "$1" = "cpu" ]; then
          python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND --cpu_num_cores='1' --cpu_memory='3.75' --cpu_memory_info='GB' --platform_type='GCP' --platform_machine_type='n1-standard-1'
      else
          python benchmarks/scripts/keras_benchmarks/run_benchmark.py --keras_backend=$KERAS_BACKEND --gpu_count='1' --gpu_platform='NVIDIA Tesla K80' --cpu_num_cores='8' --cpu_memory='30' --cpu_memory_info='GB' --platform_type='GCP' --platform_machine_type='n1-standard-8'
      fi
done

