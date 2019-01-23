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

import setup
import perfzero.common.utils as utils
import perfzero.common.perfzero_config as perfzero_config


class TestSetupRunner(unittest.TestCase):

  def setUp(self):
    self.config = perfzero_config.PerfZeroConfig(mode='mock')
    super(TestSetupRunner, self).setUp()

  def test_get_gcs_downloads(self):
    """Tests parsing gcs download environment variable."""
    path_0 = [
        'cifar10', 'gs://tf-perf-imagenet-uswest1/tensorflow/cifar10_data/*'
    ]
    path_1 = [
        'imagenet', 'gs://tf-perf-imagenet-uswest1/tensorflow/somethingelse'
    ]

    self.config.gcs_downloads_str = '{},{}'.format(';'.join(path_0),
                                                   ';'.join(path_1))

    setup_runner = setup.SetupRunner(docker_file='', config=self.config)
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

    self.config.git_repos_str = '{},{}'.format(';'.join(repo_0),
                                               ';'.join(repo_1))

    setup_runner = setup.SetupRunner(docker_file='', config=self.config)
    repo_list = setup_runner._get_git_repos()
    actual_repo_0 = repo_list[0]
    actual_repo_1 = repo_list[1]
    self.assertEqual(repo_0[0], actual_repo_0['local_path'])
    self.assertEqual(repo_0[1], actual_repo_0['url'])
    self.assertEqual(repo_0[2], actual_repo_0['branch'])
    self.assertEqual(repo_1[3], actual_repo_1['sha_hash'])
