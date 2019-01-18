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

"""Extract CPU info."""
from __future__ import print_function

import perfzero.common.utils as utils


def get_cpu_info():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.

  Returns:
    Tuple of device driver version and gpu name.
  """
  model_name = _model_name()
  core_count = _core_count()
  socket_count = _socket_count()
  cpu_info = _cpu_info()

  return model_name, socket_count, core_count, cpu_info


def _model_name():
  cmd = "cat /proc/cpuinfo | grep 'model name' | sort --unique"
  retcode, result = utils.run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    model_name_parts = lines[0].split(':')
    return model_name_parts[1].strip()
  else:
    print('Error getting cpuinfo model name: {}'.format(result))
    return ''


def _core_count():
  cmd = "cat /proc/cpuinfo | grep 'cpu cores' | sort --unique"
  retcode, result = utils.run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    core_count_parts = lines[0].split(':')
    # Cores * sockets = total cores for the system.
    core_count = int(core_count_parts[1].strip())
    total_cores = core_count * _socket_count()
    return total_cores
  else:
    print('Error getting cpuinfo core count: {}'.format(result))
    return -1


def _socket_count():
  cmd = 'grep -i "physical id" /proc/cpuinfo | sort -u | wc -l'
  retcode, result = utils.run_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    return int(lines[0])
  else:
    print('Error getting cpuinfo scocket count: {}'.format(result))
    return -1


def _cpu_info():
  cmd = 'cat /proc/cpuinfo'
  retcode, result = utils.run_command(cmd)
  if retcode == 0:
    return result
  else:
    print('Error getting cpuinfo: {}'.format(result))
    return ''
