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

"""Tests cpu module."""
import unittest

from mock import patch
from perfzero.common import cpu


class TestCpu(unittest.TestCase):

  @patch('perfzero.common.utils.run_command')
  def test_get_model_name(self, run_command_mock):
    """Tests extract the cpu model name."""
    run_command_mock.return_value = [
        0, 'model name  : Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz\n'
    ]
    model_name = cpu._model_name()
    self.assertEqual('Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz', model_name)

  @patch('perfzero.common.utils.run_command')
  def test_get_socket_count(self, run_command_mock):
    """Tests get socket count."""
    run_command_mock.return_value = [0, '2\n']
    socket_count = cpu._socket_count()
    self.assertEqual(2, socket_count)

  @patch('perfzero.common.cpu._socket_count')
  @patch('perfzero.common.utils.run_command')
  def test_get_core_count(self, run_command_mock, mock_socket):
    """Tests get number of cores."""
    run_command_mock.return_value = [0, 'cpu cores  : 6\n']
    mock_socket.return_value = 2
    core_count = cpu._core_count()
    self.assertEqual(12, core_count)

  @patch('perfzero.common.utils.run_command')
  def test_get_cpuinfo(self, run_command_mock):
    """Tests git cpuinfo echos back."""
    run_command_mock.return_value = [0, 'foo I show whatever shows up\n']
    cpuinfo = cpu._cpu_info()
    self.assertEqual('foo I show whatever shows up\n', cpuinfo)
