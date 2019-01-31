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
import types
import unittest

import benchmark
import mock
import perfzero.report.benchmark_result as benchmark_result


class TestTestRunner(unittest.TestCase):

  def tearDown(self):
    # Reset ENV VARs
    envars = ['ROGUE_TEST_CLASS', 'ROGUE_TEST_METHODS', 'ROGUE_PYTHON_PATH']
    for envar in envars:
      if os.environ.get(envar) is not None:
        del os.environ[envar]
    super(TestTestRunner, self).tearDown()

  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_load_test_class(self, mock_setup):
    """Test loading module and test class."""

    # foo.fake is not found unless foo and fake are mocked.
    sys.modules['foo'] = mock.Mock()
    sys.modules['foo.fake'] = mock.Mock()
    config = mock.Mock()
    config.test_class_str = 'foo.fake.TestClass'
    config.python_paths_str = None
    benchmark_runner = benchmark.BenchmarkRunner(config)
    class_ = benchmark_runner._instantiate_benchmark_class('/dev/null')
    self.assertIsInstance(class_.oss_report_object,
                          type(benchmark_result.BenchmarkResult()))
    mock_setup.assert_called()

  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_method_filter(self, _):
    """Tests returning methods on a class based on a filter."""
    regex_filter = 'bench.*'
    config = mock.Mock()
    config.python_paths_str = None
    benchmark_runner = benchmark.BenchmarkRunner(config)
    mock_test_class = mock.Mock()
    mock_test_class.benchmark_test = 'foo'
    methods = benchmark_runner._return_methods_to_execute(mock_test_class,
                                                          regex_filter)
    self.assertEqual('benchmark_test', methods[0])

  @mock.patch('benchmark.BenchmarkRunner._exec_benchmarks')
  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_run_benchmarks_filter(self, _, exec_bench_mock):
    """Test run benchmarks with filters argument."""
    test_class = mock.Mock()
    test_class.benchmark_foo = 'foo'
    module = mock.Mock()
    sys.modules['new_foo'] = module
    module.TestClass.return_value = test_class
    config = mock.Mock()
    config.test_class_str = 'new_foo.TestClass'
    config.benchmark_methods_str = 'filter:bench'
    benchmark_runner = benchmark.BenchmarkRunner(config)

    benchmark_runner.run_benchmark()
    arg0 = exec_bench_mock.call_args[0][0]
    self.assertEqual('benchmark_foo', arg0[0])

  @mock.patch('benchmark.BenchmarkRunner._exec_benchmarks')
  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_run_benchmarks_list_of_methods(self, _, exec_bench_mock):
    """Test run benchmarks with list of methods."""
    module = mock.Mock()
    sys.modules['new_foo'] = module
    config = mock.Mock()
    config.test_class_str = 'new_foo.TestClass'
    config.benchmark_methods_str = 'benchmark_1,benchmark_2'
    benchmark_runner = benchmark.BenchmarkRunner(config)

    benchmark_runner.run_benchmark()
    arg0 = exec_bench_mock.call_args[0][0]
    self.assertEqual(config.benchmark_methods_str.split(','), arg0)
