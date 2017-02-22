# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Provides helper functions for distributed benchmarks running on Jenkins."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import calendar
from collections import namedtuple
import logging
import os

from google.protobuf import json_format

from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import gfile


_OUTPUT_FILE_ENV_VAR = 'TF_DIST_BENCHMARK_RESULTS_FILE'
_TEST_NAME_ENV_VAR = 'TF_DIST_BENCHMARK_NAME'


# Represents a single stat_value entry where
#  - name is a string
#  - stat_value is the latency to track (for e.g. mean time per iter)
#  - iters is the number of iterations
StatEntry = namedtuple(
    'StatEntry', ['name', 'stat_value', 'iters'])


def store_data_in_json(stat_entries, timestamp, output_file=None):
  """Stores benchmark results in JSON format.

  Args:
    stat_entries: list of StatEntry objects.
    timestamp: (datetime) start time of the test run.
    output_file: if specified, writes benchmark results to output_file.
      If not specified, writes results to the file specified by
      BENCHMARK_RESULTS_FILE environment variable.

  Raises:
    ValueError: when neither output_file is passed in nor
      BENCHMARK_RESULTS_FILE is set.
  """
  test_result = test_log_pb2.TestResults(
      start_time=calendar.timegm(timestamp.timetuple()))
  if not output_file:
    if _OUTPUT_FILE_ENV_VAR not in os.environ:
      logging.warning(
          'Skipping storing json output, since we could not determine '
          'location to store results at.')
      return
    output_file = os.environ[_OUTPUT_FILE_ENV_VAR]

  if _TEST_NAME_ENV_VAR in os.environ:
    test_result.name = os.environ[_TEST_NAME_ENV_VAR]
  else:
    test_result.name = 'TestBenchmark'

  for stat_entry in stat_entries:
    test_result.entries.entry.add(
        name=stat_entry.name,
        iters=stat_entry.iters,
        wall_time=stat_entry.stat_value
    )
  json_test_results = json_format.MessageToJson(test_result)

  with gfile.Open(output_file, 'wb') as jsonfile:
    jsonfile.write(json_test_results)
