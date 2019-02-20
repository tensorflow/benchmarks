#!/bin/sh
set -e

cd $(dirname $0)

KUNGFU_SRC=$(pwd)/KungFu

KUNGFU_PRUN=${KUNGFU_SRC}/bin/kungfu-prun
if [ ! -f $KUNGFU_PRUN ]; then
    ./install-kungfu.sh
fi

np=4
$KUNGFU_PRUN \
    -np=$np \
    -timeout=10m \
    python \
    tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --num_gpus=1 \
    --batch_size=32 \
    --model=resnet50 \
    --variable_update=kungfu
