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
    if mode == 'env' and 'PERFZERO_BENCHMARK_METHODS' in os.environ:
      self.benchmark_methods_str = self._get_env_var(
          'PERFZERO_BENCHMARK_METHODS')
      self.benchmark_class_str = self._get_env_var('PERFZERO_BENCHMARK_CLASS')
      self.platform_name_str = self._get_env_var('PERFZERO_PLATFORM_NAME')
      self.system_name_str = self._get_env_var('PERFZERO_SYSTEM_NAME')
      self.test_env_str = self._get_env_var('PERFZERO_TEST_ENV')
      self.python_path_str = self._get_env_var('PERFZERO_PYTHON_PATH')
      self.git_repos_str = self._get_env_var('PERFZERO_GIT_REPOS')
      self.output_gcs_url_str = self._get_env_var('PERFZERO_OUTPUT_GCS_URL',
                                                  False)
      self.gcs_downloads_str = self._get_env_var('PERFZERO_GCS_DOWNLOADS',
                                                 False)
      self.bigquery_table_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_TABLE_NAME')
      self.bigquery_project_name_str = self._get_env_var(
          'PERFZERO_BIGQUERY_PROJECT_NAME')
      self.dockerfile_path_str = self._get_env_var('PERFZERO_DOCKERFILE_PATH',
                                                   False,
                                                   'docker/Dockerfile')
    elif mode == 'env' and 'PERFZERO_BENCHMARK_METHODS' not in os.environ:
      self.benchmark_methods_str = self._get_env_var('ROGUE_TEST_METHODS')
      self.benchmark_class_str = self._get_env_var('ROGUE_TEST_CLASS')
      self.platform_name_str = self._get_env_var(
          'ROGUE_PLATFORM', False, default='unknown')
      self.system_name_str = self._get_env_var(
          'ROGUE_PLATFORM_TYPE', False, default='unknown')
      self.test_env_str = self._get_env_var(
          'ROGUE_TEST_ENV', False, default='local')
      self.python_path_str = self._get_env_var('ROGUE_PYTHON_PATH')
      self.git_repos_str = self._get_env_var('ROGUE_GIT_REPOS')
      self.output_gcs_url_str = self._get_env_var(
          'ROGUE_CODE_DIR', False, default='')
      self.gcs_downloads_str = self._get_env_var(
          'ROGUE_GCS_DOWNLOADS', False, default='')
      self.bigquery_project_name_str = self._get_env_var(
          'ROGUE_REPORT_PROJECT', False, default='')
      self.bigquery_table_name_str = 'benchmark_results.result'

      if not self.bigquery_project_name_str:
        self.bigquery_project_name_str = 'google.com:tensorflow-performance'
        self.bigquery_table_name_str = 'benchmark_results_dev.result'

    elif mode == 'mock':
      pass

  def _get_env_var(self, key, is_required=True, default=None):
    if key in os.environ and os.environ[key]:
      return os.environ[key]

    if is_required:
      raise ValueError(
          'Environment variable {} needs to be defined'.format(key))

    return default
