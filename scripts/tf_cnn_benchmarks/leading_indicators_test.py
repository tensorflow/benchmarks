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
"""Benchmark various leading indicators CNNs.

The purpose of these tests is to test each model as a high level baseline and
to ensure the various variable_update options have not regressing. Not all
options are tested.  The tests focus on the most viable options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import logging
import os
import sys

from absl import flags
from absl.testing import absltest  # pylint: disable=unused-import
import tensorflow as tf
import benchmark_cnn
from platforms import util as platforms_util

flags.DEFINE_integer('num_batches', None,
                     'number of batches to run, excluding warmup')


class BenchmarkBase(tf.test.Benchmark):
  """Base class for all benchmarks in this file."""

  def __init__(self, output_dir=None):
    # Load default values if the benchmark is not run with absl.app.run()
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

    self.fake_data_dir = os.path.join(platforms_util.get_test_data_dir(),
                                      'fake_tf_record_data')
    self.data_dir = ('/readahead/200M/placer/prod/home/distbelief/'
                     'imagenet-tensorflow/imagenet-2012-tfrecord')
    self.output_dir = output_dir

  def _run_benchmark(self, params):
    """Run a CNN benchmark and report its results.

    Args:
      params: Params tuple, typically created by benchmark_cnn.make_params or
        benchmark_cnn.make_params_from_flags.
    """
    logging.info('Running benchmark [%s]', self._get_name())
    params = benchmark_cnn.setup(params)
    bench = benchmark_cnn.BenchmarkCNN(params)
    bench.print_info()
    stats = bench.run()
    extras = {}
    extras['examples_per_sec'] = stats.get('images_per_sec')
    if 'last_average_loss' in stats:
      extras['last_average_loss'] = stats['last_average_loss']
    if 'top_1_accuracy' in stats:
      extras['top_1_accuracy'] = stats['top_1_accuracy']
    if 'top_5_accuracy' in stats:
      extras['top_5_accuracy'] = stats['top_5_accuracy']
    self.report_benchmark(
        iters=stats.get('num_steps'),
        wall_time=stats.get('average_wall_time'),
        extras=extras)

  def _shared_params(self):
    """Returns shared parameters for all benchmarks in this file."""
    params = {}
    if flags.FLAGS.num_batches is not None:
      params['num_batches'] = flags.FLAGS.num_batches
    if self.output_dir is not None:
      params['benchmark_log_dir'] = self.output_dir
    return benchmark_cnn.make_params(**params)

  def _binary_search_batch_size(self, params, init_batch_size):
    """Find the max batch_size using binary search."""
    assert init_batch_size > 0
    low_batch_size = 0
    high_batch_size = None
    batch_size = init_batch_size

    # No need to run a warmup or many batches; if it doesn't OOM after 10
    # batches, it should work in general.
    params = params._replace(num_batches=10, num_warmup_batches=0)

    # Find high_batch_size first.
    tf.logging.info(
        'Looking for upper bound to batch size, starting with %d' % batch_size)
    while high_batch_size is None:
      tf.logging.info('Trying batch_size %d' % batch_size)
      params = params._replace(batch_size=batch_size)
      bench = benchmark_cnn.BenchmarkCNN(params)
      bench.print_info()
      try:
        bench.run()
        low_batch_size = batch_size
        batch_size *= 2
      except tf.errors.ResourceExhaustedError:
        high_batch_size = batch_size - 1

    # Binary Search
    tf.logging.info(
        'Max batch size is in range (%d, %d].  Starting binary search to find '
        'exact max batch size.' % (low_batch_size, batch_size))
    while low_batch_size < high_batch_size:
      batch_size = (low_batch_size + high_batch_size + 1) // 2
      tf.logging.info('Trying batch_size %d' % batch_size)
      params = params._replace(batch_size=batch_size)
      bench = benchmark_cnn.BenchmarkCNN(params)
      bench.print_info()
      try:
        bench.run()
        low_batch_size = batch_size
      except tf.errors.ResourceExhaustedError:
        high_batch_size = batch_size - 1
    self.report_benchmark(extras={'max_batch_size': low_batch_size})


class Resnet50BenchmarksInferenceCpu(BenchmarkBase):
  """"Benchmarks for ResNet50 inference on CPU."""

  def _shared_params(self):
    """Returns shared parameters for all ResNet50 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        num_gpus=1,
        model='resnet50',
        num_warmup_batches=5,
        num_batches=50,
        distortions=False,
        forward_only=True,
        device='cpu',
        data_format='NHWC',
        num_intra_threads=0)

  def benchmark_synth_forward_batch1(self):
    """Tests 1 CPU batch size 1."""
    params = self._shared_params()._replace(batch_size=1)
    self._run_benchmark(params)

  def benchmark_synth_forward_batch16(self):
    """Tests 1 CPU batch size 16."""
    params = self._shared_params()._replace(batch_size=16)
    self._run_benchmark(params)


class FrozenResnet50BenchmarksInferenceCpu(Resnet50BenchmarksInferenceCpu):
  """"Benchmarks for ResNet50 frozen graph inference on CPU."""

  def _shared_params(self):
    return super(FrozenResnet50BenchmarksInferenceCpu,
                 self)._shared_params()._replace(freeze_when_forward_only=True)


class Resnet50BenchmarksInference(BenchmarkBase):
  """"Benchmarks for ResNet50 inference."""

  def _shared_params(self):
    """Returns shared parameters for all ResNet50 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        num_gpus=1, model='resnet50', distortions=False, forward_only=True)

  def benchmark_synth_forward_batch128(self):
    """Tests 1 GPU batch size 128."""
    params = self._shared_params()._replace(batch_size=128)
    self._run_benchmark(params)

  def benchmark_fp16_synth_forward_batch128(self):
    """Tests 1 GPU batch size 128 FP16."""
    params = self._shared_params()._replace(batch_size=128, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_fp16_synth_forward_batch16(self):
    """Tests 1 GPU batch size 16 FP16."""
    params = self._shared_params()._replace(batch_size=16, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_synth_forward_batch128(self):
    """Tests 1 GPU batch size 128 with XLA."""
    params = self._shared_params()._replace(batch_size=128, xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_forward_batch128(self):
    """Tests 1 GPU batch size 128 FP16 with XLA."""
    params = self._shared_params()._replace(
        batch_size=128, use_fp16=True, xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_forward_batch16(self):
    """Tests 1 GPU batch size 16 FP16 with XLA."""
    params = self._shared_params()._replace(
        batch_size=16, use_fp16=True, xla=True)
    self._run_benchmark(params)


class FrozenResnet50BenchmarksInference(Resnet50BenchmarksInference):
  """"Benchmarks for ResNet50 frozen graph inference."""

  def _shared_params(self):
    return super(FrozenResnet50BenchmarksInference,
                 self)._shared_params()._replace(freeze_when_forward_only=True)

  def benchmark_trt_synth_forward_batch128(self):
    """Tests 1 GPU batch size 128."""
    params = self._shared_params()._replace(batch_size=128, trt_mode='FP32')
    self._run_benchmark(params)

  # TODO(laigd): enable fp16 tests for TF-TRT, it's currently not supported yet.
  # def benchmark_fp16_trt_synth_forward_batch128(self):
  #   """Tests 1 GPU batch size 128 FP16."""
  #   params = self._shared_params()._replace(
  #       batch_size=128, use_fp16=True, trt_mode='FP16')
  #   self._run_benchmark(params)

  # Test with batch size 16 to compare with native TF GPU implementation and
  # XLA.
  # def benchmark_fp16_trt_synth_forward_batch16(self):
  #   """Tests 1 GPU batch size 16 FP16."""
  #   params = self._shared_params()._replace(
  #       batch_size=16, use_fp16=True, trt_mode='FP16')
  #   self._run_benchmark(params)


class Resnet50Benchmarks(BenchmarkBase):
  """"Benchmark resnet50 configurations."""

  def _shared_params(self):
    """Returns shared parameters for all ResNet50 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='resnet50', batch_size=128, distortions=False)

  def _shared_params_fp16(self):
    """Returns shared parameters for all ResNet50 FP16 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='resnet50',
        batch_size=256,
        distortions=False,
        use_fp16=True,
    )

  def benchmark_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params()._replace(num_gpus=1)
    self._run_benchmark(params)

  def benchmark_synth_1gpu_gpuparams_batch64(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params()._replace(num_gpus=1, batch_size=64)
    self._run_benchmark(params)

  def benchmark_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data."""
    params = self._shared_params()._replace(
        num_gpus=1, data_dir=self.fake_data_dir, data_name='imagenet')
    self._run_benchmark(params)

  def benchmark_synth_1gpu_max_batch_size(self):
    """Finds largest batch size that can be run with 1 gpu using synth data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._binary_search_batch_size(params, init_batch_size=128)

  def benchmark_synth_4gpu_gpuparams(self):
    """Tests 4 gpus with synthetic data with parameters on the gpus."""
    params = self._shared_params()._replace(
        num_gpus=4, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fake_4gpu_gpuparams(self):
    """Tests 4 gpus with fake data with parameters on the gpus."""
    params = self._shared_params()._replace(
        num_gpus=4,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_synth_4gpu_cpuparams(self):
    """Tests 4 gpus with synthetic data with parameters on the cpu."""
    params = self._shared_params()._replace(
        num_gpus=4,
        variable_update='parameter_server',
        local_parameter_device='cpu')
    self._run_benchmark(params)

  def benchmark_synth_8gpu_cpuparams(self):
    """Tests 8 gpus with synthetic data with parameters on the cpu."""
    params = self._shared_params()._replace(
        num_gpus=8,
        variable_update='parameter_server',
        local_parameter_device='cpu')
    self._run_benchmark(params)

  def benchmark_fake_4gpu_cpuparams(self):
    """Tests 4 gpus with fake data with parameters on the cpu."""
    params = self._shared_params()._replace(
        num_gpus=4,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        local_parameter_device='cpu')
    self._run_benchmark(params)

  def benchmark_fake_8gpu_cpuparams(self):
    """Tests 8 gpus with fake data with parameters on the cpu."""
    params = self._shared_params()._replace(
        num_gpus=8,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        local_parameter_device='cpu')
    self._run_benchmark(params)

  def benchmark_synth_4gpu_gpureplicated(self):
    """Tests 4 gpu with synthetic data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=4,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_synth_8gpu_gpureplicated(self):
    """Tests 8 gpu with synthetic data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=8,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fake_8gpu_gpureplicated(self):
    """Tests 8 gpu with fake data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=8,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  # FP16 mixed-precisions tests.

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with parameters on the gpu."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams_batch128(self):
    """Tests 1 gpu with synthetic data with parameters on the gpu."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, batch_size=128, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams_batch64(self):
    """Tests 1 gpu with synthetic data with parameters on the gpu."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, batch_size=64, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_4gpu_gpuparams(self):
    """Tests 4 gpus with synthetic data with parameters on the gpus."""
    params = self._shared_params_fp16()._replace(
        num_gpus=4, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_4gpu_gpureplicated(self):
    """Tests 4 gpu with synthetic data with nccl and all_reduce."""
    params = self._shared_params_fp16()._replace(
        num_gpus=4,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fp16_synth_8gpu_gpureplicated(self):
    """Tests 8 gpu with synthetic with nccl and all_reduce."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fp16_fake_1gpu_gpuparams(self):
    """Tests 1 gpus with fake data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_fake_8gpu_gpureplicated(self):
    """Tests 8 gpus with fake data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fp16_fakedistort_8gpu_gpureplicated(self):
    """Tests 8 gpus with fake distorted data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        distortions=True,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  # XLA versions of Resnet50 tests only for single GPU.
  def benchmark_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, synthetic data with XLA."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True, use_fp16=True)
    self._run_benchmark(params)

  # Test does not run as part of continuous testing on guitar.
  def benchmark_ng_xla_batch64_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with XLA, synth data, and batch 64."""
    params = self._shared_params()._replace(
        num_gpus=1, batch_size=64, variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_batch64_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, XLA, synth data, and batch 64."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1,
        batch_size=64,
        variable_update='parameter_server',
        xla=True,
        use_fp16=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_batch128_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, XLA, and synth data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1,
        batch_size=128,
        variable_update='parameter_server',
        xla=True,
        use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_synth_1gpu_max_batch_size(self):
    """Finds largest batch that can be run with XLA, 1 gpu, and synth data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._binary_search_batch_size(params, init_batch_size=128)

  def benchmark_xla_real_1gpu_gpuparams(self):
    """Tests 1 gpu with real data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.data_dir,
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)

  # Test does not run as part of continuous testing.
  def benchmark_xla_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)

  # Test does not run as part of continuous testing.
  def benchmark_xla_fakedistort_1gpu_gpuparams(self):
    """Tests 1 gpu with fake distorted data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        distortions=True,
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)


class Resnet50v15Benchmarks(BenchmarkBase):
  """"Benchmark various ResNet50V1.5 configurations.

  ResNetV1.5 differs from V1 in stride 2 is used in the first 3x3 convolution of
  each block instead of the first 1x1 convolution.
  """

  def _shared_params_fp16(self):
    """Returns shared parameters for all ResNet50v1.5 FP16 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='resnet50_v1.5',
        batch_size=256,
        distortions=False,
        use_fp16=True,
    )

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params_fp16()._replace(num_gpus=1)
    self._run_benchmark(params)

  def benchmark_fp16_batch256_synth_8gpu_gpuparams(self):
    """Tests 8 gpus with synthetic data at batch 256."""
    params = self._shared_params_fp16()._replace(num_gpus=8, batch_size=256)
    self._run_benchmark(params)

  def benchmark_fp16_batch128_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data at batch 128 (useful for small GPUs)."""
    params = self._shared_params_fp16()._replace(num_gpus=1, batch_size=128)
    self._run_benchmark(params)

  def benchmark_fp16_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, data_dir=self.fake_data_dir, data_name='imagenet')
    self._run_benchmark(params)

  def benchmark_fp16_synth_4gpu_cpuparams(self):
    """Tests 4 gpus with synthetic data with parameters on the cpu."""
    params = self._shared_params_fp16()._replace(
        num_gpus=4,
        variable_update='parameter_server',
        local_parameter_device='cpu')
    self._run_benchmark(params)

  def benchmark_fp16_synth_8gpu_gpureplicated(self):
    """Tests 8 gpu with synthetic data with parameters replicated."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        num_batches=200,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fp16_fake_8gpu_gpureplicated(self):
    """Tests 8 gpu with fake data with parameters replicated."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        num_batches=200,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2)
    self._run_benchmark(params)

  # XLA versions of Resnet50v1.5 tests.
  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, synthetic data with XLA."""
    params = self._shared_params_fp16()._replace(num_gpus=1, xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_batch128_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, batch128, synthetic data with XLA."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, batch_size=128, xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params_fp16()._replace(num_gpus=1, xla_compile=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_batch128_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data at batch 128 (useful for small GPUs)."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1, num_batches=200, batch_size=128, xla_compile=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_batch256_synth_8gpu_gpuparams(self):
    """Tests 8 gpu with synthetic data and xla autojit."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8, num_batches=200, batch_size=256, xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data."""
    params = self._shared_params_fp16()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        xla_compile=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_synth_8gpu_gpureplicated(self):
    """Tests 8 gpu with synthetic data with parameters replicated."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        num_batches=200,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        xla_compile=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_fake_8gpu_gpureplicated(self):
    """Tests 8 gpu with fake data with parameters replicated."""
    params = self._shared_params_fp16()._replace(
        num_gpus=8,
        num_batches=200,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        xla_compile=True)
    self._run_benchmark(params)


class Vgg16Benchmarks(BenchmarkBase):
  """"Benchmark various vgg16 configurations."""

  def _shared_params(self):
    """Returns shared parameters for all vgg16 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='vgg16', batch_size=128, distortions=False)

  def benchmark_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with parameters on gpu."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with parameters on gpu."""
    params = self._shared_params()._replace(
        num_gpus=1, use_fp16=True, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_synth_4gpu_gpureplicated(self):
    """Tests 4 gpu with synthetic data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=4,
        all_reduce_spec='nccl',
        variable_update='replicated',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_synth_8gpu_gpureplicated(self):
    """Tests 8 gpu with synthetic data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=8,
        all_reduce_spec='nccl',
        variable_update='replicated',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  # XLA versions of VGG16 tests only for single GPU.
  def benchmark_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, synthetic data, and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True, use_fp16=True)
    self._run_benchmark(params)

  # Test does not run as part of continuous testing.
  def benchmark_xla_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)

  def benchmark_xla_real_1gpu_gpuparams(self):
    """Tests 1 gpu with real data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.data_dir,
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)


class TrivialBenchmarks(BenchmarkBase):
  """"Benchmarks for trivial model.

  The purpose of these tests is to verify the upper bound for the input
  pipeline. Fake data creates an upperbound on the input pipeline throughput.
  """

  def _shared_params(self):
    """Returns shared parameters for all trivial benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='trivial',
        num_gpus=8,
        distortions=False,
        variable_update='independent',
        data_dir=self.fake_data_dir)

  def benchmark_fake_64batch(self):
    params = self._shared_params()._replace(batch_size=64, data_name='imagenet')
    self._run_benchmark(params)

  def benchmark_fake_128batch(self):
    params = self._shared_params()._replace(
        batch_size=128, data_name='imagenet')
    self._run_benchmark(params)

  def benchmark_fake_256batch(self):
    params = self._shared_params()._replace(
        batch_size=256, data_name='imagenet')
    self._run_benchmark(params)

  def benchmark_fakedistort_128batch(self):
    params = self._shared_params()._replace(
        batch_size=128, data_name='imagenet', distortions=True)
    self._run_benchmark(params)


class AlexnetBenchmarks(BenchmarkBase):
  """"Benchmarks for alexnet."""

  def _shared_params(self):
    """Returns shared parameters for all alexnet benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='alexnet', batch_size=512, distortions=False)

  def benchmark_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with parameters on gpu."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data with parameters on gpu."""
    params = self._shared_params()._replace(
        num_gpus=1, use_fp16=True, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_synth_8gpu_gpureplicated(self):
    """Tests 8 gpus with synthetic data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=8,
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  def benchmark_fake_8gpu_gpureplicated(self):
    """Tests 8 gpus with fake data with parameters replicated."""
    params = self._shared_params()._replace(
        num_gpus=8,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='replicated',
        all_reduce_spec='nccl',
        compact_gradient_transfer=False,
        gradient_repacking=2)
    self._run_benchmark(params)

  # XLA Benchmark tests for AlexNet.
  def benchmark_xla_synth_1gpuparams(self):
    """Tests 1 gpu with synthetic data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, synthetic data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True, use_fp16=True)
    self._run_benchmark(params)

  # Test does not run as part of continuous testing.
  def benchmark_xla_fake_1gpuparams(self):
    """Tests 1 gpu with fake data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)

  def benchmark_xla_real_1gpuparams(self):
    """Tests 1 gpu with real data and XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.data_dir,
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)


class InceptionV3Benchmarks(BenchmarkBase):
  """"Benchmark for InceptionV3."""

  def _shared_params(self):
    """Returns shared parameters for all InceptionV3 benchmarks."""
    return BenchmarkBase._shared_params(self)._replace(
        model='inception3', batch_size=64, distortions=False)

  def benchmark_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic data."""
    params = self._shared_params()._replace(
        num_gpus=1, use_fp16=True, variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_synth_1gpu_max_batch_size(self):
    """Finds largest batch size that can be run with 1 gpu using synth data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server')
    self._binary_search_batch_size(params, init_batch_size=128)

  def benchmark_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with synthetic and  XLA."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    """Tests 1 gpu with fp16, XLA and synthetic data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_synth_1gpu_max_batch_size(self):
    """Finds largest batch that can be run with XLA, 1 gpu, and synth data."""
    params = self._shared_params()._replace(
        num_gpus=1, variable_update='parameter_server', xla=True)
    self._binary_search_batch_size(params, init_batch_size=128)

  # Test does not run as part of continuous testing.
  def benchmark_xla_fake_1gpu_gpuparams(self):
    """Tests 1 gpu with fake data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.fake_data_dir,
        data_name='imagenet',
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)

  def benchmark_xla_real_1gpu_gpuparams(self):
    """Tests 1 gpu with real data with XLA."""
    params = self._shared_params()._replace(
        num_gpus=1,
        data_dir=self.data_dir,
        variable_update='parameter_server',
        xla=True)
    self._run_benchmark(params)


class NcfBenchmarks(BenchmarkBase):
  """Benchmarks for neural collaborative filtering."""

  def _shared_params(self):
    return BenchmarkBase._shared_params(self)._replace(
        model='ncf', batch_size=64*1024, num_gpus=1, num_warmup_batches=1)

  def benchmark_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_fp16_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla=True, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_compile_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla_compile=True)
    self._run_benchmark(params)

  def benchmark_fp16_xla_compile_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla_compile=True, use_fp16=True)
    self._run_benchmark(params)


class DeepSpeech2Benchmarks(BenchmarkBase):
  """Benchmarks for DeepSpeech2 model."""

  def _shared_params(self):
    return BenchmarkBase._shared_params(self)._replace(
        model='deepspeech2', batch_size=32, num_gpus=1, data_name='librispeech')

  def benchmark_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(variable_update='parameter_server')
    self._run_benchmark(params)

  def benchmark_xla_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla=True)
    self._run_benchmark(params)

  def benchmark_xla_compile_synth_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        variable_update='parameter_server', xla_compile=True)
    self._run_benchmark(params)


class SsdBenchmarks(BenchmarkBase):
  """Benchmarks for SSD model."""

  def _cudnn_version(self):
    if sys.platform == 'win32':
      return None

    lib = ctypes.cdll.LoadLibrary(None)
    if hasattr(lib, 'cudnnGetErrorString'):
      version = lib.cudnnGetVersion()
      return version

    return None

  def _shared_params(self):
    cudnn_version = self._cudnn_version()
    if cudnn_version is None or cudnn_version < 7300:
      raise RuntimeError(
          'Needs at least cuDNN 7.3 to work with fp16 (b/112048183). '
          'Build with --define=use_experimental_cudnn=1')

    return BenchmarkBase._shared_params(self)._replace(
        # TODO(b/115672206): Replace backbone model and data dir with replicated
        # placer location for better performance.
        backbone_model_path=platforms_util.get_ssd_backborn_model_file(),  # pylint: disable=line-too-long
        data_dir=platforms_util.get_ssd_backboard_data_dir(),
        batch_size=128,
        data_name='coco',
        model='ssd300',
        num_batches=10,
        num_warmup_batches=1,
        num_gpus=1,
        optimizer='momentum',
        momentum=0.9,
        weight_decay=5e-4,
        loss_type_to_report='base_loss',
        single_l2_loss_op=True,
        compute_lr_on_cpu=True,
    )

  def benchmark_xla_compile_real_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=1,
        xla_compile=True,
    )
    self._run_benchmark(params)

  def benchmark_real_1gpu_gpuparams(self):
    params = self._shared_params()._replace(num_gpus=1,)
    self._run_benchmark(params)

  def benchmark_xla_compile_fp16_real_1gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=1, xla_compile=True, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_fp16_real_1gpu_gpuparams(self):
    params = self._shared_params()._replace(num_gpus=1, use_fp16=True)
    self._run_benchmark(params)

  def benchmark_xla_compile_real_8gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=8,
        xla_compile=True,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        num_batches=50,
    )
    self._run_benchmark(params)

  def benchmark_real_8gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=8,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        num_batches=50,
    )
    self._run_benchmark(params)

  def benchmark_xla_compile_fp16_real_8gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=8,
        xla_compile=True,
        use_fp16=True,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        num_batches=50,
    )
    self._run_benchmark(params)

  def benchmark_fp16_real_8gpu_gpuparams(self):
    params = self._shared_params()._replace(
        num_gpus=8,
        use_fp16=True,
        variable_update='replicated',
        all_reduce_spec='nccl',
        gradient_repacking=2,
        num_batches=50,
    )
    self._run_benchmark(params)


if __name__ == '__main__':
  tf.test.main()
