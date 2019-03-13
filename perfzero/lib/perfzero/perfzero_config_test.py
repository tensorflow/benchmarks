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
"""Tests for perfzero_config.py."""
from __future__ import print_function

import os
import unittest
import perfzero.perfzero_config as perfzero_config


class TestPerfZeroConfig(unittest.TestCase):

  def test_get_git_repos(self):
    config = perfzero_config.PerfZeroConfig(mode='mock')
    config.git_repos_str = 'https://github.com/tensorflow/benchmarks.git;branch_1;hash_1,https://github.com/tensorflow/models.git;branch_2'
    git_repos = config.get_git_repos('/site_package_dir')

    git_repo_1 = {}
    git_repo_1['url'] = 'https://github.com/tensorflow/benchmarks.git'
    git_repo_1['dir_name'] = 'benchmarks'
    git_repo_1['local_path'] = '/site_package_dir/benchmarks'
    git_repo_1['branch'] = 'branch_1'
    git_repo_1['git_hash'] = 'hash_1'

    git_repo_2 = {}
    git_repo_2['url'] = 'https://github.com/tensorflow/models.git'
    git_repo_2['dir_name'] = 'models'
    git_repo_2['local_path'] = '/site_package_dir/models'
    git_repo_2['branch'] = 'branch_2'

    self.assertEqual(2, len(git_repos))
    self.assertEqual(git_repo_1, git_repos[0])
    self.assertEqual(git_repo_2, git_repos[1])

  def test_get_env_vars(self):
    config = perfzero_config.PerfZeroConfig(mode='mock')
    self.assertEqual({}, config.get_env_vars())

    os.environ['PERFZERO_VAR1'] = 'value1'
    self.assertEqual({'PERFZERO_VAR1': 'value1'}, config.get_env_vars())


