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
"""PerfZero configs provided by user."""
from __future__ import print_function

import os


class PerfZeroConfig(object):
  """Creates and contains config for PerfZero."""

  def __init__(self, mode):
    # In this mode, PERFZERO_BENCHMARK_CLASS and PERFZERO_BENCHMARK_METHODS
    # are used to determine the benchmark methods to execute. This mode is
    # deprecated because it constraints one perfzero execution to only run
    # benchmark methods from one class
    if mode == 'env' and 'PERFZERO_BENCHMARK_CLASS' in os.environ:
      benchmark_methods_str = self._get_env_var(
          'PERFZERO_BENCHMARK_METHODS')
      benchmark_class_str = self._get_env_var('PERFZERO_BENCHMARK_CLASS')
      self.benchmark_methods_maybe_filter = []
      for benchmark_method in benchmark_methods_str.split(','):
        self.benchmark_methods_maybe_filter.append(benchmark_class_str + '.' +
                                                   benchmark_method)
      self.platform_name_str = self._get_env_var('PERFZERO_PLATFORM_NAME')
      self.system_name_str = self._get_env_var('PERFZERO_SYSTEM_NAME')
      self.python_path_str = self._get_env_var('PERFZERO_PYTHON_PATH')
      self.git_repos_str = self._get_env_var('PERFZERO_GIT_REPOS')
      self.output_gcs_url_str = self._get_env_var('PERFZERO_OUTPUT_GCS_URL',
                                                  False)
      self.gcs_downloads_str = self._get_env_var('PERFZERO_GCS_DOWNLOADS',
                                                 False)
      self.bigquery_dataset_table_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_TABLE_NAME', False)
      self.bigquery_project_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_PROJECT_NAME', False)
      self.dockerfile_path_str = self._get_env_var('PERFZERO_DOCKERFILE_PATH',
                                                   False, 'docker/Dockerfile')
      self.ml_framework_build_label_str = self._get_env_var(
          'PERFZERO_ML_FRAMEWORK_BUILD_LABEL', False)
      self.execution_label_str = self._get_env_var('PERFZERO_EXECUTION_LABEL',
                                                   False)
    # In this mode, environment variables with prefix PERFZERO_BENCHMARK_METHOD
    # are used to dertermine the benchmark methods we want to execute. The
    # value of these environment variables encodes both the benchmark class and
    # benchmark method name. And it allows user to run arbitrary combination of
    # benchmark methods in one perfzero execution
    elif mode == 'env' and 'PERFZERO_PLATFORM_NAME' in os.environ:
      self.benchmark_methods_maybe_filter = []
      for key in os.environ.keys():
        if key.startswith('PERFZERO_BENCHMARK_METHOD'):
          self.benchmark_methods_maybe_filter.append(os.environ[key])
      self.platform_name_str = self._get_env_var('PERFZERO_PLATFORM_NAME')
      self.system_name_str = self._get_env_var('PERFZERO_SYSTEM_NAME')
      self.test_env_str = self._get_env_var('PERFZERO_TEST_ENV')
      self.python_path_str = self._get_env_var('PERFZERO_PYTHON_PATH')
      self.git_repos_str = self._get_env_var('PERFZERO_GIT_REPOS')
      self.output_gcs_url_str = self._get_env_var('PERFZERO_OUTPUT_GCS_URL',
                                                  False)
      self.gcs_downloads_str = self._get_env_var('PERFZERO_GCS_DOWNLOADS',
                                                 False)
      self.bigquery_dataset_table_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_TABLE_NAME', False)
      self.bigquery_project_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_PROJECT_NAME', False)
      self.dockerfile_path_str = self._get_env_var('PERFZERO_DOCKERFILE_PATH',
                                                   False, 'docker/Dockerfile')
      self.ml_framework_build_label_str = self._get_env_var(
          'PERFZERO_ML_FRAMEWORK_BUILD_LABEL', False)
      self.execution_label_str = self._get_env_var('PERFZERO_EXECUTION_LABEL',
                                                   False)
    elif mode == 'mock':
      pass

  def get_env_vars(self):
    env_vars = {}
    for key in os.environ.keys():
      if key.startswith('PERFZERO_'):
        env_vars[key] = os.environ[key]
    return env_vars

  def _get_env_var(self, key, is_required=True, default=None):
    if key in os.environ and os.environ[key]:
      return os.environ[key]
    if is_required:
      raise ValueError(
          'Environment variable {} needs to be defined'.format(key))
    return default
