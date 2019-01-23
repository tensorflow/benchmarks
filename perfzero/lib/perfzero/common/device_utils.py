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

"""Setup the data drive with raid, RAM, or mount network drives."""
from __future__ import print_function

import perfzero.common.utils as utils


def get_nvme_devices():
  """Returns list paths to nvme devices."""
  devices = []
  cmd = 'sudo lsblk'
  retcode, log = utils.run_command(cmd)
  if retcode:
    raise Exception('"{}" failed with code:{} and log:\n{}'.format(
        cmd, retcode, log))

  lines = log.splitlines()
  if lines:
    for line in lines:
      if line.startswith('nvme'):
        parts = line.split()
        devices.append('/dev/' + parts[0].strip())
  return devices


def create_gce_nvme_raid(data_dir, list_of_devices):
  """Creates a raid zero array of nvme drives."""

  cmd = 'sudo mountpoint -q {}'.format(data_dir)
  retcode, _ = utils.run_command(cmd)
  if retcode:
    cmds = []
    # GCE nvme drives some times are in an odd state and
    # think they are in another raid. mdadm doe snot have -y option.
    # or the kokoro images were left dirty? and that is where the info
    # comes from.
    cmds.append('yes | sudo mdadm --create /dev/md0 --level=0 '
                '--raid-devices={} {}'.format(
                    len(list_of_devices), ' '.join(list_of_devices)))
    cmds.append('sudo mkfs.ext4 -F /dev/md0')
    cmds.append('sudo mkdir -p {}'.format(data_dir))
    cmds.append('sudo mount /dev/md0 {}'.format(data_dir))
    cmds.append('sudo chmod a+w {}'.format(data_dir))

    utils.run_commands(cmds)
    print('Created and mounted RAID array at {}'.format(data_dir))
  else:
    print('Skipping RAID array creation since path {} already exists'.format(
        data_dir))


def create_ram_disk(data_dir, disk_size):
  """Create a RAM disk."""

  cmd = 'sudo mountpoint -q {}'.format(data_dir)
  retcode, _ = utils.run_command(cmd)
  if retcode:
    cmds = []
    cmds.append('sudo mkdir -p {}'.format(data_dir))
    cmds.append('sudo mount -t tmpfs -o size={}m tmpfs {}'.format(
        disk_size, data_dir))

    utils.run_commands(cmds)
  else:
    print('RAM disk or something else is mounted at {}'.format(data_dir))
