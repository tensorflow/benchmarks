#!/bin/bash

/usr/local/bin/pip install --user tf-nightly
python -m tf_cnn_benchmarks.tf_cnn_benchmarks
