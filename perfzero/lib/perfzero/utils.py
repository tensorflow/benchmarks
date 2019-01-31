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

import os
import re
import subprocess
import sys
import time
import logging


def checkout_git_repo(url, local_path, branch=None, sha_hash=None):
  """Clone, update, or synce a repo.

  If the clone already exists the repo will be updated via a pull.

  Args:
    url (str): Git repo url
    local_path (str): Local path to place the repo
    branch (str, optional): Branch to checkout.
    sha_hash (str, optional): Hash to reset to.
  """
  if os.path.isdir(local_path):
    git_clone_or_pull = 'git -C {} pull'.format(local_path)
  else:
    git_clone_or_pull = 'git clone {} {}'.format(url, local_path)
  run_command(git_clone_or_pull)

  if branch is not None:
    branch_cmd = 'git -C {} checkout {}'.format(local_path, branch)
    run_command(branch_cmd)

  if sha_hash is not None:
    sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(local_path, sha_hash)
    run_command(sync_to_hash_cmd)

  logging.info('Checked out repo from {} to {}'.format(url, local_path))


def setup_python_path(site_packages_dir, python_path_str):
  if python_path_str:
    python_paths = python_path_str.split(',')
    for python_path in python_paths:
      sys.path.append(os.path.join(site_packages_dir, python_path))
  logging.debug('PYTHONPATH: {}'.format(sys.path))


def active_gcloud_service(auth_token_path):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token_path
  cmds = [
      'gcloud auth activate-service-account --key-file {}'.format(
          auth_token_path)
  ]
  run_commands(cmds)


def download_from_gcs(gcs_path, local_path):
  make_dir_if_not_exist(local_path)

  cmds = [['gsutil', '-m', 'cp', '-r', '-n', gcs_path, local_path]]
  run_commands(cmds, shell=False)
  logging.info('Downloaded data from gcs {} to local directory {}'.format(
      gcs_path, local_path))


def upload_to_gcs(local_dir, output_gcs_dir):
  cmds = ['gsutil -m cp -r {} {}'.format(local_dir, output_gcs_dir)]
  run_commands(cmds)
  logging.info('Uploaded data from local directory {} to gcs {}'.format(
      local_dir, output_gcs_dir))


def make_dir_if_not_exist(local_path):
  if not os.path.exists(local_path):
    os.makedirs(local_path)
    logging.info('Created directory {}'.format(local_path))


def run_command(cmd, shell=True):
  """Structures for a variety of different test results.

  Args:
    cmd: Command to execute
    shell: True to use shell, false otherwise.

  Returns:
    Tuple of the command return value and the standard out in as a string.
  """
  logging.debug('Execute command: {}'.format(cmd))
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
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
    logging.error('Error getting cpuinfo model name: {}'.format(result))
    return ''


def get_cpu_core_count():
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
    logging.error('Error getting cpuinfo core count: {}'.format(result))
    return -1


def get_cpu_socket_count():
  cmd = 'grep -i "physical id" /proc/cpuinfo | sort -u | wc -l'
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    return int(lines[0])
  else:
    logging.error('Error getting cpuinfo scocket count: {}'.format(result))
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
    logging.error('nvidia-smi did not return as expected:{}'.format(result))
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
  from google.protobuf import json_format
  from tensorflow.core.util import test_log_pb2

  if not os.path.isfile(benchmark_result_file_path):
    logging.error(
        'Failed to read benchmark result because file {} does not exist'.format(
            benchmark_result_file_path))
    return {}

  with open(benchmark_result_file_path, 'rb') as f:
    benchmark_entries = test_log_pb2.BenchmarkEntries()
    benchmark_entries.ParseFromString(f.read())

    return json_format.MessageToDict(
        benchmark_entries, preserving_proto_field_name=True)['entry'][0]
