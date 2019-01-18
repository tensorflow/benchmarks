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

"""Tests for test_runner script."""
from __future__ import print_function

import os
import sys
import unittest

import benchmark
from mock import Mock
import perfzero.report.bench_result as bench_result


class TestTestRunner(unittest.TestCase):

  def tearDown(self):
    # Reset ENV VARs
    envars = ['ROGUE_TEST_CLASS', 'ROGUE_TEST_METHODS', 'ROGUE_PYTHON_PATH']
    for envar in envars:
      if os.environ.get(envar) is not None:
        del os.environ[envar]
    super(TestTestRunner, self).tearDown()

  @unittest.skip('')
  def test_load_test_class(self):
    """Tests ok_to_run not finding existing processes."""
    sys.modules['foo.fake'] = Mock()
    os.environ['ROGUE_TEST_CLASS'] = 'foo.fake.TestClass'
    os.environ['ROGUE_TEST_METHODS'] = 'TestMethod'
    os.environ['ROGUE_PYTHON_PATH'] = 'models'
    test_runner_ = benchmark.BenchmarkRunner()
    class_ = test_runner_._load_test_class('/dev/null')
    self.assertIsInstance(class_.oss_report_object,
                          type(bench_result.BenchResult()))
