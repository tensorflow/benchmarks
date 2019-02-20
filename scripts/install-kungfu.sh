#!/bin/sh
set -e

cd $(dirname $0)

KUNGFU_SRC=$(pwd)/KungFu

install_kungfu() {
    local KUNGFU_GIT_URL=git@github.com:lsds/KungFu.git
    if [ ! -d ${KUNGFU_SRC} ]; then
        git clone ${KUNGFU_GIT_URL} ${KUNGFU_SRC}
    fi
    cd ${KUNGFU_SRC}
    ./scripts/azure/gpu-machine/install-golang1.11.sh
    export PATH=$HOME/local/go/bin:$PATH
    ./scripts/go-install.sh --build-gtest
    pip install --no-index -U .
    cd -
}

install_kungfu
