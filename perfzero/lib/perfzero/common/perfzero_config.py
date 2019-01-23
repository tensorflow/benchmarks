"""PerfZero configs provided by user"""
from __future__ import print_function

import os

import perfzero.common.utils as utils


class PerfZeroConfig(object):

  def __init__(self, mode):
    if mode == 'env':
      self.gcs_downloads_str = self._get_env_var('ROGUE_GCS_DOWNLOADS', '')
      self.git_repos_str = self._get_env_var('ROGUE_GIT_REPOS')
      self.project_name_str = self._get_env_var(
          'ROGUE_REPORT_PROJECT', default='LOCAL')
      self.benchmark_methods_str = self._get_env_var('ROGUE_TEST_METHODS')
      self.test_env_str = self._get_env_var('ROGUE_TEST_ENV', default='local')
      self.platform_name_str = self._get_env_var(
          'ROGUE_PLATFORM', default='unknown')
      self.platform_type_str = self._get_env_var(
          'ROGUE_PLATFORM_TYPE', default='unknown')
      self.output_gcs_url_str = self._get_env_var('ROGUE_CODE_DIR', default='')
      self.test_class_str = self._get_env_var('ROGUE_TEST_CLASS')
      self.python_paths_str = self._get_env_var('ROGUE_PYTHON_PATH')
    elif mode == 'mock':
      pass

  def _get_env_var(self, key, default=None):
    if key in os.environ:
      return os.environ[key]
    if default is not None:
      return default

    raise ValueError('Environment variable {} needs to be defined'.format(key))
