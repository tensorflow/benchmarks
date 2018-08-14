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
"""Tests for result_upload."""
import json
import unittest

import mock
import result_info
import result_upload


class TestResultUpload(unittest.TestCase):

  def test_build_row(self):
    """Tests creating row to be inserted.

    Test is too big but currently test aspects of `result_info` along with
    creating the final row with `result_upload`.
    """
    test_result, results = result_info.build_test_result(
        'fake_test_id',
        123.4,
        test_harness='unit_tests',
        test_environment='unit_test_env')
    system_info = result_info.build_system_info(
        platform='aws', platform_type='p3.8xlarge')
    test_info = result_info.build_test_info(batch_size=32)

    mock_user = 'foo.boyd@nope'
    credentials = mock.Mock()
    credentials = mock.Mock(service_account_email=mock_user)
    row = result_upload._build_row(
        credentials,
        test_result,
        results,
        system_info=system_info,
        test_info=test_info)

    test_info_actual = json.loads(row['test_info'])
    system_info_actual = json.loads(row['system_info'])
    result_info_actual = json.loads(row['result_info'])

    # Verifies core columns.
    self.assertIn('timestamp', row)
    self.assertIn('result_id', row)
    self.assertEqual(mock_user, row['user'])
    self.assertEqual('fake_test_id', row['test_id'])
    self.assertEqual('unit_tests', row['test_harness'])
    self.assertEqual('unit_test_env', row['test_environment'])

    # Verifies system_info.
    self.assertEqual('aws', system_info_actual['platform'])
    self.assertEqual('p3.8xlarge', system_info_actual['platform_type'])

    # Verifies test_info.
    self.assertEqual(32, test_info_actual['batch_size'])

    # Verifies result_info for primary (first) result.
    self.assertEqual(123.4, result_info_actual[0]['result'])
    self.assertEqual('total_time', result_info_actual[0]['result_type'])
    self.assertEqual('ms', result_info_actual[0]['result_units'])
