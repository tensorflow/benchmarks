#!/bin/sh
set -e
cd $(dirname $0)/..

KUNGFU_GIT_URL=git@github.com:lsds/KungFu.git
if [ -d scripts/KungFu ]; then
    rm -fr scripts/KungFu
fi
if [ ! -d scripts/KungFu ]; then
    git clone $KUNGFU_GIT_URL scripts/KungFu
fi
docker build -f docker/Dockerfile.hvd_tf_benchmarks -t hvd-bench:latest .
