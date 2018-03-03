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
"""Provides a way to store benchmark results in GCE Datastore.

Datastore client is initialized from current environment.
Data is stored using the format defined in:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/test/upload_test_benchmarks_index.yaml
"""
from datetime import datetime
import json
import os
import sys

import six

from google.cloud import datastore


_TEST_NAME_ENV_VAR = 'TF_DIST_BENCHMARK_NAME'


def upload_to_benchmark_datastore(data, test_name=None, start_time=None):
  """Use a new datastore.Client to upload data to datastore.

  Create the datastore Entities from that data and upload them to the
  datastore in a batch using the client connection.

  Args:
    data: Map from benchmark names to values.
    test_name: Name of this test. If not specified, name will be set either
      from TF_DIST_BENCHMARK_NAME environment variable or to default name
      'TestBenchmark'.
    start_time: (datetime) Time to record for this test.

  Raises:
    ValueError: if test_name is not passed in and TF_DIST_BENCHMARK_NAME
      is not set.
  """
  client = datastore.Client()

  if not test_name:
    if _TEST_NAME_ENV_VAR in os.environ:
      test_name = os.environ[_TEST_NAME_ENV_VAR]
    else:
      raise ValueError(
          'No test name passed in for benchmarks. '
          'Either pass a test_name to upload_to_benchmark_datastore or '
          'set %s environment variable.' % _TEST_NAME_ENV_VAR)
  test_name = six.text_type(test_name)

  if not start_time:
    start_time = datetime.now()

  # Create one Entry Entity for each benchmark entry.  The wall-clock timing is
  # the attribute to be fetched and displayed.  The full entry information is
  # also stored as a non-indexed JSON blob.
  entries = []
  batch = []
  for name, value in data.items():
    e_key = client.key('Entry')
    e_val = datastore.Entity(e_key, exclude_from_indexes=['info'])
    entry_map = {'name': name, 'wallTime': value, 'iters': '1'}
    entries.append(entry_map)
    e_val.update({
        'test': test_name,
        'start': start_time,
        'entry': six.text_type(name),
        'timing': value,
        'info': six.text_type(json.dumps(entry_map))
    })
    batch.append(e_val)

  # Create the Test Entity containing all the test information as a
  # non-indexed JSON blob.
  test_result = json.dumps(
      {'name': test_name,
       'startTime': (start_time - datetime(1970, 1, 1)).total_seconds(),
       'entries': {'entry': entries},
       'runConfiguration': {'argument': sys.argv[1:]}})
  t_key = client.key('Test')
  t_val = datastore.Entity(t_key, exclude_from_indexes=['info'])
  t_val.update({
      'test': test_name,
      'start': start_time,
      'info': six.text_type(test_result)
  })
  batch.append(t_val)

  # Put the whole batch of Entities in the datastore.
  client.put_multi(batch)
