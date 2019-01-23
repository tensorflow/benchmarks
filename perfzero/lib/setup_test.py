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

"""Tests data disk config module."""
from __future__ import print_function

import os
import unittest

import perfzero.common.utils as utils
import setup


class TestSetupRunner(unittest.TestCase):

  def tearDown(self):
    # Reset ENV VARs
    envars = [
        'ROGUE_CODE_DIR', 'ROGUE_ZERO_PLATFORM_TYPE', 'ROGUE_REPORT_PROJECT',
        'ROGUE_GIT_REPOS', 'ROGUE_TEST_METHODS', 'ROGUE_TEST_CLASS',
        'ROGUE_PYTHON_PATH'
    ]
    for envar in envars:
      if os.environ.get(envar) is not None:
        del os.environ[envar]
    super(TestSetupRunner, self).tearDown()

  def test_get_gcs_download_list(self):
    """Tests parsing gcs download environment variable."""
    path_0 = [
        'cifar10', 'gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/*'
    ]
    path_1 = [
        'imagenet', 'gs://tf-perf-imagenet-uswest1/tensorflow/somethingelse'
    ]

    gcs_envar = '{},{}'.format(';'.join(path_0), ';'.join(path_1))

    os.environ['ROGUE_GCS_DOWNLOADS'] = gcs_envar

    setup_runner = setup.SetupRunner('')
    gcs_list = setup_runner._get_gcs_downloads()
    actual_gcs_0 = gcs_list[0]
    actual_gcs_1 = gcs_list[1]
    self.assertEqual(path_0[0], actual_gcs_0['local_path'])
    self.assertEqual(path_0[1], actual_gcs_0['gcs_path'])
    self.assertEqual(path_1[1], actual_gcs_1['gcs_path'])

  def test_get_git_repos_list(self):
    """Tests parsing repos environment variables."""
    repo_0 = ['models', 'https://github.com/tensorflow/models.git', 'master']
    repo_1 = [
        'models', 'https://github.com/tensorflow/models.git', 'master',
        'sha-hash'
    ]

    repo_envar = '{},{}'.format(';'.join(repo_0), ';'.join(repo_1))

    os.environ['ROGUE_GIT_REPOS'] = repo_envar

    setup_runner = setup.SetupRunner('')
    repo_list = setup_runner._get_git_repos()
    actual_repo_0 = repo_list[0]
    actual_repo_1 = repo_list[1]
    self.assertEqual(repo_0[0], actual_repo_0['local_path'])
    self.assertEqual(repo_0[1], actual_repo_0['url'])
    self.assertEqual(repo_0[2], actual_repo_0['branch'])
    self.assertEqual(repo_1[3], actual_repo_1['sha_hash'])

  def test_env_var_check_and_print(self):
    os.environ['ROGUE_TEST_METHODS'] = 'test_method_1,test_method_2'
    os.environ['ROGUE_TEST_CLASS'] = 'foo.bar.Class'
    os.environ['ROGUE_PYTHON_PATH'] = 'models'
    os.environ['ROGUE_GIT_REPOS'] = ('models;'
                                     'https://github.com/tensorflow/models.git;'
                                     'master')
    os.environ['ROGUE_ZERO_PLATFORM_TYPE'] = '8xV100'

    setup_runner = setup.SetupRunner('')
    utils.check_and_print_env_var()

  def test_env_var_check_and_print_missing(self):
    os.environ['ROGUE_TEST_METHODS'] = 'test_method_1,test_method_2'
    os.environ['ROGUE_TEST_CLASS'] = 'foo.bar.Class'

    setup_runner = setup.SetupRunner('')
    with self.assertRaises(ValueError):
      utils.check_and_print_env_var()
