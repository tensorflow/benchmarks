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


class TestBenchmarkRunner(unittest.TestCase):

  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_load_benchmark_class(self, mock_setup):
    """Test loading module and test class."""

    # foo.fake is not found unless foo and fake are mocked.
    sys.modules['foo'] = mock.Mock()
    sys.modules['foo.fake'] = mock.Mock()
    config = mock.Mock()
    config.benchmark_class_str = 'foo.fake.TestClass'
    config.python_paths_str = None
    benchmark_runner = benchmark.BenchmarkRunner(config)
    class_instance = benchmark_runner._instantiate_benchmark_class('/dev/null')
    mock_setup.assert_called()

  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_get_benchmark_methods_filter(self, mock_setup):
    """Tests returning methods on a class based on a filter."""
    config = mock.Mock()
    config.python_paths_str = None
    config.benchmark_methods_str = 'filter:bench.*'
    config.benchmark_class_str = 'new_foo.BenchmarkClass'
    benchmark_runner = benchmark.BenchmarkRunner(config)

    mock_benchmark_class = mock.Mock()
    mock_benchmark_class.benchmark_method_1 = 'foo'

    mock_module = mock.Mock()
    sys.modules['new_foo'] = mock_module
    mock_module.BenchmarkClass.return_value = mock_benchmark_class

    methods = benchmark_runner._get_benchmark_methods()

    self.assertEqual(1, len(methods))
    self.assertEqual('benchmark_method_1', methods[0])

  @mock.patch('benchmark.BenchmarkRunner._setup')
  def test_get_benchmark_methods_exact_match(self, mock_setup):
    """Tests returning methods on a class based on a filter."""
    config = mock.Mock()
    config.python_paths_str = None
    config.benchmark_methods_str = 'benchmark_method_1,benchmark_method_2'
    config.benchmark_class_str = 'new_foo.BenchmarkClass'
    benchmark_runner = benchmark.BenchmarkRunner(config)

    methods = benchmark_runner._get_benchmark_methods()
    self.assertEqual(['benchmark_method_1', 'benchmark_method_2'], methods)
