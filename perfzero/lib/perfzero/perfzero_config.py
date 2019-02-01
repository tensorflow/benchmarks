"""PerfZero configs provided by user"""
from __future__ import print_function

import os

import perfzero.utils as utils


class PerfZeroConfig(object):

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

      if self.bigquery_project_name_str == '':
        self.bigquery_project_name_str = 'google.com:tensorflow-performance'
        self.bigquery_table_name_str = 'benchmark_results_dev.result'

    elif mode == 'mock':
      pass

  def _get_env_var(self, key, is_required=True, default=None):
    if key in os.environ:
      return os.environ[key]

    if is_required:
      raise ValueError(
          'Environment variable {} needs to be defined'.format(key))

    return default
