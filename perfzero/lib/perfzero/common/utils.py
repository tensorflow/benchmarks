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
import tensorflow as tf


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

  print('Checked out repo from {} to {}'.format(url, local_path))


def setup_python_path(library_dir):
  module_paths_str = get_env_var('ROGUE_PYTHON_PATH', default=None)
  if module_paths_str:
    module_paths = module_paths_str.split(',')
    for module_path in module_paths:
      sys.path.append(os.path.join(library_dir, module_path))
  print('PYTHONPATH:{}'.format(sys.path))


def active_gcloud_service(auth_token_path):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token_path
  cmds = [
      'gcloud auth activate-service-account --key-file {}'.format(
          auth_token_path)
  ]
  run_commands(cmds)


def download_from_gcs(gcs_path, local_path):
  make_dir_if_not_exist(local_path)

  # Splits command into parts due to '-m cp -r'.
  cmds = [['gsutil', '-m', 'cp', '-r', '-n', gcs_path, local_path]]
  run_commands(cmds, shell=False)
  print('Downloaded data from {} to directory {}'.format(gcs_path, local_path))


def get_milliseconds_diff(start_time):
  """Convert seconds to int milliseconds."""
  return int(round((time.time() - start_time) * 1000))


def make_dir_if_not_exist(local_path):
  if not os.path.exists(local_path):
    os.makedirs(local_path)
    print('Created directory {}'.format(local_path))


def run_command(cmd, shell=True):
  """Structures for a variety of different test results.

  Args:
    cmd: Command to execute
    shell: True to use shell, false otherwise.

  Returns:
    Tuple of the command return value and the standard out in as a string.
  """
  print('Execute command: {}'.format(cmd))
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
  stdout = ''
  while True:
    retcode = p.poll()
    line = p.stdout.readline()
    line_str = line.decode('utf-8')
    print(line_str)
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


def get_env_var(varname, default=None):
  env_var_val = None
  if varname in os.environ:
    env_var_val = os.environ[varname]
  if env_var_val:
    return env_var_val
  else:
    if default:
      return default
    else:
      raise ValueError('ENV VAR was not found:{}'.format(varname))


def check_and_print_env_var():
  """Prints ENV_VARS and errors on missing required values."""
  optional_envars = [
      'ROGUE_CODE_DIR', 'ROGUE_PLATFORM_TYPE', 'ROGUE_PLATFORM',
      'ROGUE_REPORT_PROJECT', 'ROGUE_TEST_ENV'
  ]

  required_envars = [
      'ROGUE_TEST_METHODS', 'ROGUE_TEST_CLASS', 'ROGUE_PYTHON_PATH'
  ]

  print('Optional ENVIRONMENTAL VARIABLES:')
  for envar in optional_envars:
    envar_val = get_env_var(envar, default='not set')
    print('{}={}'.format(envar, envar_val))

  print('Required ENVIRONMENTAL VARIABLES:')
  for envar in required_envars:
    envar_val = get_env_var(envar)
    print('{}={}'.format(envar, envar_val))


def get_tf_full_version():
  """Returns TensorFlow version as reported by TensorFlow.

    Note: The __git__version__ can be confusing as the TensorFlow version
    number in the string often points to an older branch due to git merges.
    The git hash is still correct.  The best option is to use the numeric
    version from __version__ and the hash from __git_version__.

  Returns:
    Tuple of __version__, __git_version__
  """
  return tf.__version__, tf.__git_version__


def get_gpu_info():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.

  Returns:
    Tuple of device driver version and gpu name.
  """
  cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv'
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    gpu_info = lines[1].split(',')
    if 'Quadro' in gpu_info[1] and len(lines) > 2:
      gpu_info = lines[2].split(',')
      return gpu_info[0].strip(), gpu_info[1].strip()
    else:
      return gpu_info[0].strip(), gpu_info[1].strip()
  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return '', ''


def get_gpu_count():
  cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv'
  retcode, result = run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    return len(lines) - 1
  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return -1


def get_running_processes():
  """Returns list of `dict` objects representing running processes on GPUs."""
  retcode, result = run_command('nvidia-smi')
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    # Goes to the first line with the word Processes, jumps down one and then
    # parses the list of processes.
    look_for_processes = False
    processes = []
    for line in lines:
      # Summary line starts with images/sec
      if line.find('Processes') > 0:
        look_for_processes = True

      if look_for_processes:
        p = re.compile('[0-1]+')
        m = p.search(line)
        if m and m.span()[0] == 5:
          line_parts = line.strip().replace('|', '').split()
          processes.append(line_parts)

    return processes

  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return '', ''


def is_ok_to_run():
  """Returns true if the system is free to run GPU tests.

  Checks the list of processes and if the list is empty or if the list of
  processes does not contain actual ML jobs returns true.  Non-ML Jobs
  like 'Xorg' or even 'cinnamon' are not a problem.

  Note: Currently this method returns true if no python processes are found.
    Which seems more sane than a long list of processes that do not matter. On
    clean systems the process list should be and normally is zero.
  """
  processes = get_running_processes()
  for process in processes:
    # Checks process name position for process named 'python'.
    if process[3].lower().find('python') > 0:
      return False
  return True
