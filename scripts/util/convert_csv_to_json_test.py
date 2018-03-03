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
"""Tests for convert_csv_to_json."""
import csv
import datetime
import unittest

import convert_csv_to_json


class ConvertCsvToJsonTest(unittest.TestCase):

  def testSingleEntryCSV(self):
    # Description,timestamp,num_batches,time mean value,time sd
    csv_reader = csv.reader(
        ['abc,2017-06-26 02:59:29.325579,10,2.15,0.1'])
    timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(csv_reader)
    self.assertEqual(
        datetime.datetime(2017, 6, 26, 2, 59, 29, 325579),
        timestamp)
    self.assertEqual(1, len(stat_entries))
    self.assertEqual('abc', stat_entries[0].name)
    self.assertEqual(2.15, stat_entries[0].stat_value)

  def testTwoEntryCSV(self):
    # Description,timestamp,num_batches,time mean value,time sd
    csv_reader = csv.reader(
        ['abc,2017-06-26 02:59:35.425579,10,2.15,0.1',
         'def,2017-06-26 02:59:29.325579,10,10.1,0.1'])
    timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(csv_reader)
    self.assertEqual(
        datetime.datetime(2017, 6, 26, 2, 59, 35, 425579),
        timestamp)
    self.assertEqual(2, len(stat_entries))
    self.assertEqual('abc', stat_entries[0].name)
    self.assertEqual(2.15, stat_entries[0].stat_value)
    self.assertEqual('def', stat_entries[1].name)
    self.assertEqual(10.1, stat_entries[1].stat_value)

  def testInvalidCSV_LessEntries(self):
    csv_reader = csv.reader(
        ['abc,2017-06-26 02:59:29.325579,10,2.15'])
    with self.assertRaises(ValueError):
      timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(
          csv_reader)

  def testInvalidCSV_MoreEntries(self):
    csv_reader = csv.reader(
        ['abc,2017-06-26 02:59:29.325579,10,2.15,0.1,extra_entry'])
    with self.assertRaises(ValueError):
      timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(
          csv_reader)

  def testInvalidCSV_EmptyEntry(self):
    csv_reader = csv.reader(
        [',2017-06-26 02:59:29.325579,10,2.15,0.1'])
    with self.assertRaises(ValueError):
      timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(
          csv_reader)

  def testInvalidCSV_InvalidDate(self):
    csv_reader = csv.reader(['abc,invaliddate,10,2.15,0.1'])
    with self.assertRaises(ValueError):
      timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(
          csv_reader)

  def testInvalidCSV_InvalidValue(self):
    csv_reader = csv.reader(
        ['abc,2017-06-26 02:59:29.325579,10,invalidfloat,0.1'])
    with self.assertRaises(ValueError):
      timestamp, stat_entries = convert_csv_to_json.get_data_from_csv(
          csv_reader)


if __name__ == '__main__':
  unittest.main()
