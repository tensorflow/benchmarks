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
#  - name is a unique identifier for this specific measurement.
#  - stat_value is the measurement to track (for e.g. mean time per iter).
#  - num_samples is the number of samples that stat_value is averaged over.
StatEntry = namedtuple(
    'StatEntry', ['name', 'stat_value', 'num_samples'])


def store_data_in_json(
    stat_entries, timestamp, output_file=None, test_name=None):
  """Stores benchmark results in JSON format.

  Args:
    stat_entries: list of StatEntry objects.
    timestamp: (datetime) start time of the test run.
    output_file: if specified, writes benchmark results to output_file.
      Otherwise, if TF_DIST_BENCHMARK_RESULTS_FILE environment variable is set,
      writes to file specified by this environment variable. If neither
      output_file is passed in, nor TF_DIST_BENCHMARK_RESULTS_FILE is set,
      does nothing.
    test_name: benchmark name. This argument is required if
      TF_DIST_BENCHMARK_NAME environment variable is not set.

  Raises:
    ValueError: when neither test_name is passed in nor
      TF_DIST_BENCHMARK_NAME is set.
  """
  test_result = test_log_pb2.TestResults(
      start_time=calendar.timegm(timestamp.timetuple()))
  if not output_file:
    if _OUTPUT_FILE_ENV_VAR not in os.environ:
      logging.warning(
          'Skipping storing json output, since we could not determine '
          'location to store results at. Either output_file argument or '
          '%s environment variable needs to be set.', _OUTPUT_FILE_ENV_VAR)
      return
    output_file = os.environ[_OUTPUT_FILE_ENV_VAR]

  if test_name is not None:
    test_result.name = test_name
  elif _TEST_NAME_ENV_VAR in os.environ:
    test_result.name = os.environ[_TEST_NAME_ENV_VAR]
  else:
    raise ValueError(
        'Could not determine test name. test_name argument is not passed in '
        'and TF_DIST_BENCHMARK_NAME environment variable is not set.')

  for stat_entry in stat_entries:
    test_result.entries.entry.add(
        name=stat_entry.name,
        iters=stat_entry.num_samples,
        wall_time=stat_entry.stat_value
    )
  json_test_results = json_format.MessageToJson(test_result)

  with gfile.Open(output_file, 'wb') as jsonfile:
    jsonfile.write(json_test_results)
