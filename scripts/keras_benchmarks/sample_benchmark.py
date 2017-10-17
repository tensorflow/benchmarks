# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sample Benchmark file."""

import time
import tensorflow as tf

import cifar10_cnn_benchmark
import mnist_mlp_benchmark

# Define a class that extends from tf.test.Benchmark.
class RunKerasBenchmarks():

  # Note: benchmark method name must start with `benchmark`.
  def benchmarkSum(self):
    with tf.Session() as sess:
      x = tf.constant(10)
      y = tf.constant(5)
      result = tf.add(x, y)

      iters = 100
      start_time = time.time()
      for _ in range(iters):
        sess.run(result)
      total_wall_time = time.time() - start_time

      print("total_wall_time", total_wall_time)



keras_benchmarks = RunKerasBenchmarks()
keras_benchmarks.benchmarkSum()

cifar = cifar10_cnn_benchmark.Cifar10CnnBenchmark()
cifar.benchmarkCifar10Cnn()

mnist = mnist_mlp_benchmark.MnistMlpBenchmark()
mnist.benchmarkMnistMlp()
