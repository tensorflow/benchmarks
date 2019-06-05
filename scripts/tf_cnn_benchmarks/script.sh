#!/usr/bin/env bash

name_models='alexnet resnet20 resnet_v2 resnet32 resnet32_v2 resnet44 resnet44_v2 resnet56 resnet56_v2 resnet110 resnet110_v2 trivial nasnet'

for model in $name_models
do
python3 tf_cnn_benchmarks.py --num_gpus=$0 --batch_size=32 --model=$name --variable_update=parameter_server
done