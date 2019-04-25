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
"""Tests utils.py."""

import os
import unittest
from mock import call
from mock import MagicMock
from mock import patch
import perfzero.utils as utils
import tensorflow as tf  # pylint: disable=g-bad-import-order


class TestUtils(unittest.TestCase, tf.test.Benchmark):

  def test_protobuf_read(self):
    output_dir = '/tmp/'
    os.environ['TEST_REPORT_FILE_PREFIX'] = output_dir
    benchmark_result_file_path = os.path.join(output_dir,
                                              'TestUtils.testReportBenchmark')
    if os.path.exists(benchmark_result_file_path):
      os.remove(benchmark_result_file_path)

    self.report_benchmark(
        iters=2000,
        wall_time=1000,
        name='testReportBenchmark',
        metrics=[{'name': 'metric_name_1', 'value': 0, 'min_value': 1},
                 {'name': 'metric_name_2', 'value': 90, 'min_value': 0,
                  'max_value': 95}])

    actual_result = utils.read_benchmark_result(
        benchmark_result_file_path)
    os.remove(benchmark_result_file_path)

    expected_result = {
        'name': 'TestUtils.testReportBenchmark',
        # google.protobuf.json_format.MessageToDict() will convert
        # int64 field to string.
        'iters': '2000',
        'wall_time': 1000,
        'cpu_time': 0,
        'throughput': 0,
        'extras': {},
        'metrics': [
            {
                'name': 'metric_name_1',
                'value': 0,
                'min_value': 1
            },
            {
                'name': 'metric_name_2',
                'value': 90,
                'min_value': 0,
                'max_value': 95
            }
        ]
    }

    self.assertDictEqual(expected_result, actual_result)

  @patch('perfzero.utils.get_git_repo_info')
  @patch('perfzero.utils.run_commands')
  def test_checkout_git_repos(self, run_commands_mock, get_git_repo_info_mock):
    git_repo_1 = {}
    git_repo_1['url'] = 'url_1'
    git_repo_1['local_path'] = 'local_path_1'
    git_repo_1['dir_name'] = 'dir_name_1'
    git_repo_1['branch'] = 'branch_1'
    git_repo_1['git_hash'] = 'git_hash_1'

    git_repo_2 = {}
    git_repo_2['url'] = 'url_2'
    git_repo_2['local_path'] = 'local_path_2'
    git_repo_2['dir_name'] = 'dir_name_2'
    git_repo_2['branch'] = 'branch_2'

    git_repo_info_1 = {'url': 'url_1'}
    git_repo_info_2 = {'url': 'url_2'}
    get_git_repo_info_mock.side_effect = \
        lambda local_path: git_repo_info_1 if local_path == 'local_path_1' else git_repo_info_2  # pylint: disable=line-too-long
    site_package_info = utils.checkout_git_repos([git_repo_1, git_repo_2],
                                                 False)

    self.assertEqual(2, len(site_package_info))
    self.assertEqual(git_repo_info_1, site_package_info['dir_name_1'])
    self.assertEqual(git_repo_info_2, site_package_info['dir_name_2'])

    run_commands_mock.assert_has_calls(any_order=False, calls=[
        call(['git clone url_1 local_path_1']),
        call(['git -C local_path_1 checkout branch_1']),
        call(['git -C local_path_1 pull --rebase']),
        call(['git -C local_path_1 reset --hard git_hash_1']),
        call(['git clone url_2 local_path_2']),
        call(['git -C local_path_2 checkout branch_2'])
    ])

  @patch('perfzero.utils.run_command')
  def test_get_git_repo_info(self, run_command_mock):
    run_command_mock.side_effect = [
        [0, 'git_url'],
        [0, 'branch_name'],
        [0, 'git_hash']
    ]

    git_repo_info = utils.get_git_repo_info('local_path_1')
    self.assertEqual(
        {'url': 'git_url', 'branch': 'branch_name', 'hash': 'git_hash'},
        git_repo_info)
    run_command_mock.assert_has_calls(any_order=False, calls=[
        call('git -C local_path_1 config --get remote.origin.url'),
        call('git -C local_path_1 rev-parse --abbrev-ref HEAD'),
        call('git -C local_path_1 rev-parse HEAD')
    ])

  @patch('builtins.open')
  @patch('perfzero.utils.make_dir_if_not_exist')
  @patch('requests.get')
  @patch('perfzero.utils.run_commands')
  def test_download_data(self, run_commands_mock, requests_get_mock,
                         make_dir_mock, open_mock):  # pylint: disable=unused-argument
    get_mock = MagicMock()
    get_mock.content = 'content'
    requests_get_mock.return_value = get_mock

    download_info_1 = {'url': 'gs://remote_path_1/name_1',
                       'local_path': 'local_path_1/modified_name_1'}
    download_info_2 = {'url': 'http://remote_path_2/name_2',
                       'local_path': 'local_path_2/modified_name_2'}
    utils.download_data([download_info_1, download_info_2])

    make_dir_mock.assert_has_calls(any_order=False, calls=[
        call('local_path_1'),
        call('local_path_2')
    ])
    requests_get_mock.assert_called_once_with('http://remote_path_2/name_2',
                                              allow_redirects=True)
    run_commands_mock.assert_has_calls(any_order=False, calls=[
        call([['gsutil', '-m', 'cp', '-r', '-n',
               'gs://remote_path_1/name_1', 'local_path_1']],
             shell=False),
        call(['mv local_path_1/name_1 local_path_1/modified_name_1']),
        call(['mv local_path_2/name_2 local_path_2/modified_name_2'])
    ])

  def test_parse_data_downloads_str(self):
    data_downloads_str = 'url_1;relative_path_1,url_2;relative_path_2'
    download_infos = utils.parse_data_downloads_str('/root_data_dir',
                                                    data_downloads_str)
    self.assertEqual(2, len(download_infos))
    self.assertEqual(download_infos[0],
                     {'url': 'url_1',
                      'local_path': '/root_data_dir/relative_path_1'})
    self.assertEqual(download_infos[1],
                     {'url': 'url_2',
                      'local_path': '/root_data_dir/relative_path_2'})

  @patch('perfzero.utils.run_command')
  def test_get_cpu_name(self, run_command_mock):
    """Tests extract the cpu model name."""
    run_command_mock.return_value = [
        0, 'model name  : Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz\n'
    ]
    cpu_name = utils.get_cpu_name()
    self.assertEqual('Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz', cpu_name)

  @patch('perfzero.utils.run_command')
  def test_get_cpu_socket_count(self, run_command_mock):
    """Tests get socket count."""
    run_command_mock.return_value = [0, '2\n']
    cpu_socket_count = utils.get_cpu_socket_count()
    self.assertEqual(2, cpu_socket_count)

  @patch('perfzero.utils.run_command')
  def test_get_gpu_model(self, run_command_mock):
    # Tests get gpu info parses expected value into expected components.
    run_command_mock.return_value = [
        0, 'driver_version, name\n381.99, GTX 1080 \n'
    ]
    gpu_model = utils.get_gpu_info()['gpu_model']
    self.assertEqual('GTX 1080', gpu_model)

    # Tests gpu info returns second entry if first entry is a Quadro.
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    gpu_model = utils.get_gpu_info()['gpu_model']
    self.assertEqual('GTX 1080', gpu_model)

  @patch('perfzero.utils.run_command')
  def test_get_gpu_count(self, run_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    gpu_count = utils.get_gpu_info()['gpu_count']
    self.assertEqual(2, gpu_count)










