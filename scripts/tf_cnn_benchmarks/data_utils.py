# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""tf.data utility methods.

Collection of utility methods that make CNN benchmark code use tf.data easier.
"""
import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile


def build_prefetch_input_processing(batch_size, data_point_shape, num_splits,
                                    preprocess_fn, cpu_device, params,
                                    gpu_devices, data_type, dataset):
  """"Returns FunctionBufferingResources that do image pre(processing)."""
  with tf.device(cpu_device):
    if params.eval:
      subset = 'validation'
    else:
      subset = 'train'

    function_buffering_resources = []
    remote_fn, args = minibatch_fn(
        batch_size=batch_size,
        data_point_shape=data_point_shape,
        num_splits=num_splits,
        preprocess_fn=preprocess_fn,
        dataset=dataset,
        subset=subset,
        train=(not params.eval),
        cache_data=params.cache_data,
        num_threads=params.datasets_num_private_threads)
    for device_num in range(len(gpu_devices)):
      with tf.device(gpu_devices[device_num]):
        buffer_resource_handle = prefetching_ops.function_buffering_resource(
            f=remote_fn,
            output_types=[data_type, tf.int32],
            target_device=cpu_device,
            string_arg=args[0],
            buffer_size=params.datasets_prefetch_buffer_size,
            shared_name=None)
        function_buffering_resources.append(buffer_resource_handle)
    return function_buffering_resources


def build_multi_device_iterator(batch_size, num_splits, preprocess_fn,
                                cpu_device, params, gpu_devices, dataset):
  """Creates a MultiDeviceIterator."""
  assert num_splits == len(gpu_devices)
  with tf.name_scope('batch_processing'):
    if params.eval:
      subset = 'validation'
    else:
      subset = 'train'
    batch_size_per_split = batch_size // num_splits
    ds = create_dataset(
        batch_size,
        num_splits,
        batch_size_per_split,
        preprocess_fn,
        dataset,
        subset,
        train=(not params.eval),
        cache_data=params.cache_data,
        num_threads=params.datasets_num_private_threads)
    multi_device_iterator = prefetching_ops.MultiDeviceIterator(
        ds,
        gpu_devices,
        source_device=cpu_device,
        max_buffer_size=params.multi_device_iterator_max_buffer_size)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         multi_device_iterator.initializer)
    return multi_device_iterator


def get_inputs_and_labels(function_buffering_resource, data_type):
  """Given a FunctionBufferingResource obtains images and labels from it."""
  return prefetching_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffering_resource,
      output_types=[data_type, tf.int32])


def create_dataset(batch_size,
                   num_splits,
                   batch_size_per_split,
                   preprocess_fn,
                   dataset,
                   subset,
                   train,
                   cache_data,
                   num_threads=None):
  """Creates a dataset for the benchmark."""
  glob_pattern = dataset.tf_record_pattern(subset)
  file_names = gfile.Glob(glob_pattern)
  if not file_names:
    raise ValueError('Found no files in --data_dir matching: {}'
                     .format(glob_pattern))
  ds = tf.data.TFRecordDataset.list_files(file_names)
  ds = ds.apply(
      interleave_ops.parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=10))
  if cache_data:
    ds = ds.take(1).cache().repeat()
  counter = tf.data.Dataset.range(batch_size)
  counter = counter.repeat()
  ds = tf.data.Dataset.zip((ds, counter))
  ds = ds.prefetch(buffer_size=batch_size)
  if train:
    ds = ds.shuffle(buffer_size=10000)
  ds = ds.repeat()
  ds = ds.apply(
      batching.map_and_batch(
          map_func=preprocess_fn,
          batch_size=batch_size_per_split,
          num_parallel_batches=num_splits))
  ds = ds.prefetch(buffer_size=num_splits)
  if num_threads:
    ds = threadpool.override_threadpool(
        ds,
        threadpool.PrivateThreadPool(
            num_threads, display_name='input_pipeline_thread_pool'))
  return ds


def create_iterator(ds):
  ds_iterator = ds.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, ds_iterator.initializer)
  return ds_iterator


def minibatch_fn(batch_size, data_point_shape, num_splits, preprocess_fn,
                 dataset, subset, train, cache_data, num_threads):
  """Returns a function and list of args for the fn to create a minibatch."""
  batch_size_per_split = batch_size // num_splits
  with tf.name_scope('batch_processing'):
    ds = create_dataset(batch_size, num_splits, batch_size_per_split,
                        preprocess_fn, dataset, subset, train, cache_data,
                        num_threads)
    ds_iterator = create_iterator(ds)

    ds_iterator_string_handle = ds_iterator.string_handle()

    @function.Defun(tf.string)
    def _fn(h):
      remote_iterator = tf.data.Iterator.from_string_handle(
          h, ds_iterator.output_types, ds_iterator.output_shapes)
      labels, inputs = remote_iterator.get_next()
      inputs = tf.reshape(
          inputs, shape=[batch_size_per_split] + data_point_shape)
      labels = tf.reshape(labels, [batch_size_per_split])
      return inputs, labels

    return _fn, [ds_iterator_string_handle]
