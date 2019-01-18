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
import perfzero.common.utils as utils


class TestUtils(unittest.TestCase):

  @patch('perfzero.common.utils.run_command')
  def test_get_gpu_info(self, run_command_mock):
    """Tests get gpu info parses expected value into expected components."""
    run_command_mock.return_value = [0, 'blah blah\n381.99, GTX 1080 \n']
    driver, gpu_info = utils.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)

  @patch('perfzero.common.utils.run_command')
  def test_get_gpu_info_quadro(self, run_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    driver, gpu_info = utils.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)

  @patch('perfzero.common.utils.run_command')
  def test_get_gpu_count(self, run_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    gpu_count = utils.get_gpu_count()
    self.assertEqual(2, gpu_count)

  @patch('perfzero.common.utils.run_command')
  def test_is_ok_to_run_false(self, run_command_mock):
    """Tests ok_to_run finding existing processes."""
    smi_test = 'perfzero/common/test_files/example_nvidia-smi_processes.txt'
    with open(smi_test) as f:
      run_command_mock.return_value = [0, f.read()]
    ok_to_run = utils.is_ok_to_run()
    self.assertFalse(ok_to_run)

  @patch('perfzero.common.utils.run_command')
  def test_is_ok_to_run(self, run_command_mock):
    """Tests ok_to_run not finding existing processes."""
    smi_test = 'perfzero/common/test_files/example_nvidia-smi_no_processes.txt'
    with open(smi_test) as f:
      run_command_mock.return_value = [0, f.read()]
    ok_to_run = utils.is_ok_to_run()
    self.assertTrue(ok_to_run)
