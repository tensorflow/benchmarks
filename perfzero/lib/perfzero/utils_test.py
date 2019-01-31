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
"""Tests untils module."""
import unittest

from mock import patch
import perfzero.utils as utils


class TestUtils(unittest.TestCase):

  @patch('perfzero.utils.run_command')
  def test_get_gpu_info(self, run_command_mock):
    """Tests get gpu info parses expected value into expected components."""
    run_command_mock.return_value = [
        0, 'driver_version, name\n381.99, GTX 1080 \n'
    ]
    gpu_model = utils.get_gpu_info()['gpu_model']
    self.assertEqual('GTX 1080', gpu_model)

  @patch('perfzero.utils.run_command')
  def test_get_gpu_info_quadro(self, run_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    gpu_model = utils.get_gpu_info()['gpu_model']
    self.assertEqual('GTX 1080', gpu_model)

  @patch('perfzero.utils.run_command')
  def test_get_gpu_count(self, run_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    gpu_count = utils.get_gpu_info()['gpu_count']
    self.assertEqual(2, gpu_count)

  @patch('perfzero.utils.run_command')
  def test_get_cpu_name(self, run_command_mock):
    """Tests extract the cpu model name."""
    run_command_mock.return_value = [
        0, 'model name  : Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz\n'
    ]
    cpu_name = utils.get_cpu_name()
    self.assertEqual('Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz', cpu_name)

  @patch('perfzero.utils.run_command')
  def test_get_cpu_socket_count(self, run_command_mock):
    """Tests get socket count."""
    run_command_mock.return_value = [0, '2\n']
    cpu_socket_count = utils.get_cpu_socket_count()
    self.assertEqual(2, cpu_socket_count)

  @patch('perfzero.utils.get_cpu_socket_count')
  @patch('perfzero.utils.run_command')
  def test_get_cpu_core_count(self, run_command_mock,
                              get_cpu_socket_count_mock):
    """Tests get number of cores."""
    run_command_mock.return_value = [0, 'cpu cores  : 6\n']
    get_cpu_socket_count_mock.return_value = 2
    cpu_core_count = utils.get_cpu_core_count()
    self.assertEqual(12, cpu_core_count)
