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
"""Git helper."""
from __future__ import print_function

import logging
import os
import subprocess
import sys


def checkout_git_repos(git_repos, force_update):
  """Clone, update, or sync a repo.

  Args:
    git_repos: array of dict containing attributes of the git repo to checkout
    force_update: Always do git pull if True

  Returns:
    A dict containing attributes of the git repositories
  """
  site_package_info = {}
  for repo in git_repos:
    if not os.path.isdir(repo['local_path']):
      run_commands(['git clone {} {}'.format(repo['url'], repo['local_path'])])
    if 'branch' in repo:
      run_commands(['git -C {} checkout {}'.format(
          repo['local_path'], repo['branch'])])
    if force_update or 'git_hash' in repo:
      run_commands(['git -C {} pull'.format(repo['local_path'])])
    if 'git_hash' in repo:
      run_commands(['git -C {} reset --hard {}'.format(
          repo['local_path'], repo['git_hash'])])
    logging.info('Checked-out repo from %s to %s',
                 repo['url'], repo['local_path'])
    site_package_info[repo['dir_name']] = get_git_repo_info(repo['local_path'])

  return site_package_info


def get_git_repo_info(local_path):
  """Get information of the git repository specified by the local_path."""
  git_repo_info = {}

  # Get git url
  cmd = 'git -C {} config --get remote.origin.url'.format(local_path)
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    git_repo_info['url'] = lines[0]
  else:
    logging.error('Error getting git url for repository %s due to %s',
                  local_path, result)
    return {}

  # Get git branch
  cmd = 'git -C {} rev-parse --abbrev-ref HEAD'.format(local_path)
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    git_repo_info['branch'] = lines[0]
  else:
    logging.error('Error getting git branch for repository %s due to %s',
                  local_path, result)
    return {}

  # Get git hash
  cmd = 'git -C {} rev-parse HEAD'.format(local_path)
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    git_repo_info['hash'] = lines[0]
  else:
    logging.error('Error getting git hash for repository %s due to %s',
                  local_path, result)
    return {}

  return git_repo_info


def setup_python_path(site_packages_dir, python_path_str):
  if python_path_str:
    python_paths = python_path_str.split(',')
    for python_path in python_paths:
      sys.path.append(os.path.join(site_packages_dir, python_path))
  logging.debug('PYTHONPATH: %s', sys.path)


def active_gcloud_service(auth_token_path):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token_path
  run_commands(['gcloud auth activate-service-account --key-file {}'.format(
      auth_token_path)])


def setup_gsutil_credential():
  run_commands(['gcloud config set pass_credentials_to_gsutil true'])


def download_from_gcs(gcs_downloads):
  """Download data from gcs_url to local_path.

  If gcs_url points to a file, then local_path will be the file with the same
  content. If gcs_url points to a directory, then local path will be the
  directory with the same content. The basename of the local_path may be
  different from the base name of the gcs_url

  Args:
    gcs_downloads: array of dict which specifies the gcs_url and local_path for
                   every downloads
  """
  for info in gcs_downloads:
    if os.path.exists(info['local_path']):
      continue
    original_base_name = os.path.basename(info['gcs_url'])
    expected_base_name = os.path.basename(info['local_path'])
    local_path_parent = os.path.dirname(info['local_path'])

    make_dir_if_not_exist(local_path_parent)
    # Download data to the local disk
    cmd = ['gsutil', '-m', 'cp', '-r', '-n', info['gcs_url'], local_path_parent]
    run_commands([cmd], shell=False)
    # Move data to the expected local path
    if original_base_name != expected_base_name:
      run_commands(['mv {} {}'.format(
          os.path.join(local_path_parent, original_base_name),
          os.path.join(local_path_parent, expected_base_name))])

    logging.info('Downloaded data from gcs %s to local path %s',
                 info['gcs_url'], info['local_path'])


def maybe_upload_to_gcs(local_dir, output_gcs_url):
  if not output_gcs_url:
    logging.info('Skipped uploading output because output_gcs_url is not set.')
    return
  run_commands(['gsutil -m cp -r {} {}'.format(local_dir, output_gcs_url)])
  logging.info('Uploaded data from local directory %s to gcs %s',
               local_dir, output_gcs_url)


def make_dir_if_not_exist(local_path):
  if not os.path.exists(local_path):
    os.makedirs(local_path)
    logging.info('Created directory %s', local_path)


def run_command(cmd, shell=True):
  """Structures for a variety of different test results.

  Args:
    cmd: Command to execute
    shell: True to use shell, false otherwise.

  Returns:
    Tuple of the command return value and the standard out in as a string.
  """
  logging.debug('Execute command: %s', cmd)
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, shell=shell)
  stdout = ''
  while True:
    retcode = p.poll()
    line = p.stdout.readline()
    line_str = line.decode('utf-8')
    logging.debug(line_str)
    stdout += line_str
    if retcode is not None:
      return retcode, stdout


def run_commands(cmds, shell=True):
  """Runs list of command and throw error if any fail."""
  for cmd in cmds:
    retcode, stdout = run_command(cmd, shell=shell)
    if retcode:
      raise Exception('"{}" failed with code:{} and stdout:\n{}'.format(
          cmd, retcode, stdout))


def get_cpu_name():
  cmd = "cat /proc/cpuinfo | grep 'model name' | sort --unique"
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    model_name_parts = lines[0].split(':')
    return model_name_parts[1].strip()
  else:
    logging.error('Error getting cpuinfo model name: %s', result)
    return ''


def get_cpu_core_count():
  """Get cpu core number."""
  cmd = "cat /proc/cpuinfo | grep 'cpu cores' | sort --unique"
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    core_count_parts = lines[0].split(':')
    # Cores * sockets = total cores for the system.
    core_count = int(core_count_parts[1].strip())
    total_cores = core_count * get_cpu_socket_count()
    return total_cores
  else:
    logging.error('Error getting cpuinfo core count: %s', result)
    return -1


def get_cpu_socket_count():
  cmd = 'grep -i "physical id" /proc/cpuinfo | sort -u | wc -l'
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    return int(lines[0])
  else:
    logging.error('Error getting cpuinfo scocket count: %s', result)
    return -1


def get_gpu_info():
  """Returns gpu information using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.

  Returns:
    A dict containing gpu_driver_version, gpu_model and gpu_count
  """
  cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv'
  retcode, result = run_command(cmd)

  if retcode != 0:
    logging.error('nvidia-smi did not return as expected: %s', result)
    return {}

  lines = result.splitlines()
  gpu_info_line = lines[1]
  if 'Quadro' in gpu_info_line and len(lines) >= 3:
    gpu_info_line = lines[2]

  gpu_info = {}
  gpu_info['gpu_driver_version'] = gpu_info_line.split(',')[0].strip()
  gpu_info['gpu_model'] = gpu_info_line.split(',')[1].strip()
  gpu_info['gpu_count'] = len(lines) - 1

  return gpu_info


def read_benchmark_result(benchmark_result_file_path):
  """Read benchmark result from the protobuf file."""
  from google.protobuf import json_format  # pylint: disable=g-import-not-at-top
  from tensorflow.core.util import test_log_pb2  # pylint: disable=g-import-not-at-top

  if not os.path.isfile(benchmark_result_file_path):
    logging.error('Failed to read benchmark result because '
                  'file %s does not exist', benchmark_result_file_path)
    return {}

  with open(benchmark_result_file_path, 'rb') as f:
    benchmark_entries = test_log_pb2.BenchmarkEntries()
    benchmark_entries.ParseFromString(f.read())

    return json_format.MessageToDict(
        benchmark_entries, preserving_proto_field_name=True)['entry'][0]
