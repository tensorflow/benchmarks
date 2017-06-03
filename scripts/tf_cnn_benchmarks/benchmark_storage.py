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
"""Provides ways to store benchmark output."""


def store_benchmark(data, storage_type=None):
  """Store benchmark data.

  Args:
    data: Dictionary mapping from string benchmark name to
      numeric benchmark value.
    storage_type: (string) Specifies where to store benchmark
      result. If storage_type is
      'cbuild_benchmark_datastore': store outputs in our continuous
        build datastore. gcloud must be setup in current environment
        pointing to the project where data will be added.
  """
  if storage_type == 'cbuild_benchmark_datastore':
    try:
      # pylint: disable=g-import-not-at-top
      import cbuild_benchmark_storage
      # pylint: enable=g-import-not-at-top
    except ImportError:
      raise ImportError(
          'Missing cbuild_benchmark_storage.py required for '
          'benchmark_cloud_datastore option')
    cbuild_benchmark_storage.upload_to_benchmark_datastore(data)
  else:
    assert False, 'unknown storage_type: ' + storage_type
