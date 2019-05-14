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
"""PerfZero utility methods."""
from __future__ import print_function

import importlib
import logging
import os
import subprocess
import sys
import threading
import traceback
import requests


def checkout_git_repos(git_repos, use_cached_site_packages):
  """Clone, update, or sync a repo.

  Args:
    git_repos: array of dict containing attributes of the git repo to checkout
    use_cached_site_packages: If true, skip git pull if the git_repo already exists

  Returns:
    A dict containing attributes of the git repositories
  """
  site_package_info = {}
  for repo in git_repos:
    logging.info('Checking out repository from %s to %s',
                 repo['url'], repo['local_path'])
    if not os.path.isdir(repo['local_path']):
      run_commands(['git clone {} {}'.format(repo['url'], repo['local_path'])])
    if 'branch' in repo:
      run_commands(['git -C {} checkout {}'.format(
          repo['local_path'], repo['branch'])])
    if not use_cached_site_packages or 'git_hash' in repo:
      run_commands(['git -C {} pull --rebase'.format(repo['local_path'])])
    if 'git_hash' in repo:
      run_commands(['git -C {} reset --hard {}'.format(
          repo['local_path'], repo['git_hash'])])
    logging.info('Checked-out repository from %s to %s',
                 repo['url'], repo['local_path'])
    site_package_info[repo['dir_name']] = get_git_repo_info(repo['local_path'])

  return site_package_info


def get_git_repo_info(local_path):
  """Get information of the git repository specified by the local_path."""
  git_repo_info = {}

  # Get git url
  cmd = 'git -C {} config --get remote.origin.url'.format(local_path)
  exit_code, result = run_command(cmd)
  lines = result.splitlines()
  if exit_code == 0 and lines:
    git_repo_info['url'] = lines[0]
  else:
    logging.error('Error getting git url for repository %s due to %s',
                  local_path, result)
    return {}

  # Get git branch
  cmd = 'git -C {} rev-parse --abbrev-ref HEAD'.format(local_path)
  exit_code, result = run_command(cmd)
  lines = result.splitlines()
  if exit_code == 0 and lines:
    git_repo_info['branch'] = lines[0]
  else:
    logging.error('Error getting git branch for repository %s due to %s',
                  local_path, result)
    return {}

  # Get git hash
  cmd = 'git -C {} rev-parse HEAD'.format(local_path)
  exit_code, result = run_command(cmd)
  lines = result.splitlines()
  if exit_code == 0 and lines:
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


def active_gcloud_service(gcloud_key_file_url, workspace_dir,
                          download_only=False):
  """Download key file and setup gcloud service credential using the key file.

  Args:
    gcloud_key_file_url: gcloud key file url
    workspace_dir: directory that the key file is downloaded to
    download_only: skip setting up the gcloud service credential if this is true
  """

  if not gcloud_key_file_url:
    return

  local_path = os.path.join(workspace_dir,
                            os.path.basename(gcloud_key_file_url))
  if not os.path.exists(local_path):
    download_data([{'url': gcloud_key_file_url, 'local_path': local_path}])

  if not download_only:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_path
    run_commands(['gcloud auth activate-service-account --key-file {}'.format(
        local_path)])
    logging.info('Activated gcloud service account credential')


def setup_gsutil_credential():
  run_commands(['gcloud config set pass_credentials_to_gsutil true'])


def download_data(download_infos):
  """Download data from url to local_path for each (url, local_path) pair in the download_infos.

  Each url should start with either gs://, http:// or https://
  Downloaded file whose name ends with .gz will be decompressed in its
  current directory

  Args:
    download_infos: array of dict which specifies the url and local_path for
                    data download
  """
  for info in download_infos:
    if os.path.exists(info['local_path']):
      continue
    original_base_name = os.path.basename(info['url'])
    expected_base_name = os.path.basename(info['local_path'])
    local_path_parent = os.path.dirname(info['local_path'])

    logging.info('Downloading data from %s to %s',
                 info['url'], info['local_path'])
    make_dir_if_not_exist(local_path_parent)
    # Download data to the local path
    if info['url'].startswith('http://') or info['url'].startswith('https://'):
      request = requests.get(info['url'], allow_redirects=True)
      f = open(info['local_path'], 'wb')
      f.write(request.content)
      f.close()
    elif info['url'].startswith('gs://'):
      cmd = ['gsutil', '-m', 'cp', '-r', '-n', info['url'], local_path_parent]
      run_commands([cmd], shell=False)
    elif info['url'].startswith('file://'):
      cmd = ['cp', info['url'][7:], local_path_parent]
      run_commands([cmd], shell=False)
    else:
      raise ValueError('Url {} with prefix {} is not supported.'.format(
          info['url'], info['url'].split(':')[0]))
    # Move data to the expected local path
    if original_base_name != expected_base_name:
      run_commands(['mv {} {}'.format(
          os.path.join(local_path_parent, original_base_name),
          os.path.join(local_path_parent, expected_base_name))])
    logging.info('Downloaded data from %s to %s',
                 info['url'], info['local_path'])
    # Decompress file if file name ends with .gz
    if info['url'].endswith('.gz'):
      run_commands(['tar xvf {} -C {}'.format(
          info['local_path'], local_path_parent)])
      logging.info('Decompressed file %s', info['local_path'])


def parse_data_downloads_str(root_data_dir, data_downloads_str):
  """Parse a comma separated string into array of dict which specifies url and local_path for every downloads.

  Args:
    root_data_dir: the directory which should contain all the dataset files
    data_downloads_str: a comma separated string specified by the
                        flag --data_downloads

  Returns:
    An array of dict which specifies the url and local_path for data download
  """

  download_infos = []
  if not data_downloads_str:
    return download_infos

  for entry in data_downloads_str.split(','):
    info = {}
    if ';' in entry:
      info['url'] = entry.split(';')[0]
      info['local_path'] = os.path.join(root_data_dir, entry.split(';')[1])
    else:
      info['url'] = entry
      info['local_path'] = os.path.join(root_data_dir, os.path.basename(entry))
    # Canonicalize url to remove trailing '/' and '*'
    if info['url'].endswith('*'):
      info['url'] = info['url'][:-1]
    if info['url'].endswith('/'):
      info['url'] = info['url'][:-1]

    download_infos.append(info)

  return download_infos


def maybe_upload_to_gcs(local_dir, output_gcs_url):
  if not output_gcs_url:
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
  logging.debug('Executing command: {}'.format(cmd))
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, shell=shell)

  exit_code = None
  line = ''
  stdout = ''
  while exit_code is None or line:
    exit_code = p.poll()
    line = p.stdout.readline().decode('utf-8')
    stdout += line
    logging.debug(line)

  return exit_code, stdout


def run_commands(cmds, shell=True):
  """Runs list of command and throw error if any fail."""
  for cmd in cmds:
    exit_code, stdout = run_command(cmd, shell=shell)
    if exit_code:
      raise Exception('"{}" failed with code:{} and stdout:\n{}'.format(
          cmd, exit_code, stdout))


def get_cpu_name():
  cmd = "cat /proc/cpuinfo | grep 'model name' | sort --unique"
  exit_code, result = run_command(cmd)
  lines = result.splitlines()
  if exit_code == 0 and lines:
    model_name_parts = lines[0].split(':')
    return model_name_parts[1].strip()
  else:
    logging.error('Error getting cpuinfo model name: %s', result)
    return ''


def get_cpu_socket_count():
  cmd = 'grep -i "physical id" /proc/cpuinfo | sort -u | wc -l'
  exit_code, result = run_command(cmd)
  lines = result.splitlines()
  if exit_code == 0 and lines:
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
  exit_code, result = run_command(cmd)

  if exit_code != 0:
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
        benchmark_entries,
        preserving_proto_field_name=True,
        including_default_value_fields=True)['entry'][0]


def print_thread_stacktrace():
  print('Here is the stacktrace for all threads:')
  thread_names = {t.ident: t.name for t in threading.enumerate()}
  for thread_id, frame in sys._current_frames().items():  # pylint: disable=protected-access
    print('Thread {}'.format(thread_names.get(thread_id, thread_id)))
    traceback.print_stack(frame)


def instantiate_benchmark_class(benchmark_class, output_dir, root_data_dir):
  """Return initialized benchmark class."""
  module_import_path, class_name = benchmark_class.rsplit('.', 1)
  module = importlib.import_module(module_import_path)
  class_ = getattr(module, class_name)
  instance = class_(output_dir=output_dir, root_data_dir=root_data_dir)

  return instance

