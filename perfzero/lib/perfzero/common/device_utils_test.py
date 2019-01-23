# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tests data disk config module."""
from __future__ import print_function

import unittest

from mock import patch
import perfzero.common.device_utils as device_utils


class TestDiskUtils(unittest.TestCase):

  @patch('perfzero.common.device_utils.get_nvme_devices')
  def test_get_device_list(self, mock_device_list):
    device_test = 'perfzero/common/test_files/nvme_device_log.txt'
    with open(device_test) as f:
      mock_device_list.return_value = f.read()
    devices = device_utils.get_nvme_devices()
    print(devices)
    self.assertTrue(devices)

