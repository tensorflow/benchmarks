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
"""Convert CSV benchmark data to JSON format.

CSV benchmark data has the format:
  Description,timestamp,num_batches,time mean value,time sd

JSON benchmark data in in the format of TestResults proto
converted to JSON.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto.
"""
import argparse
import csv
from datetime import datetime

import benchmark_util


def get_data_from_csv(csv_reader):
  """Creates a list of StatEntry objects based on data in CSV data.

  Input CSV data must be in the format:
    Description,timestamp,num_batches,time mean value,time sd

  Args:
    csv_reader: csv.reader instance.

  Returns:
    A tuple of datetime timestamp and list of benchmark_util.StatEntry objects.

  Raises:
    ValueError: if CSV is invalid.
  """
  timestamp = None
  stat_entries = []

  for row in csv_reader:
    if len(row) != 5:
      raise ValueError('Expected 5 entries per line in the input CSV file, '
                       'but found %d entries.' % len(row))
    if '' in row:
      raise ValueError('Found empty entries in row: %s' % row)

    # Set timestamp based on the first line in CSV file.
    if timestamp is None:
      # Example of time formatting: 2017-06-26 02:59:29.325579
      timestamp = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f")
    stat_entries.append(
        benchmark_util.StatEntry(row[0], float(row[3]), 1))
  return timestamp, stat_entries


def main():
  with open(FLAGS.input_csv_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    timestamp, stat_entries = get_data_from_csv(csv_reader)
    benchmark_util.store_data_in_json(
        stat_entries, timestamp,
        output_file=FLAGS.output_json_file,
        test_name=FLAGS.test_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register(
        'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
    parser.add_argument(
        '--test_name', type=str, default=None, required=True,
        help='Name of the test.')
    parser.add_argument(
        '--input_csv_file', type=str, default=None, required=True,
        help='Path to the CSV file.')
    parser.add_argument(
        '--output_json_file', type=str, default=None, required=True,
        help='Path to output JSON file.')
    FLAGS, _ = parser.parse_known_args()
    main()

