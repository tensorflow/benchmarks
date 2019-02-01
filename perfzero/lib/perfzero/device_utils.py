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

import perfzero.utils as utils
import logging


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


def create_drive_from_devices(data_dir, devices):
  """Creates a drive at data_dir based on number of devices passed."""
  cmd = 'sudo mountpoint -q {}'.format(data_dir)
  retcode, _ = utils.run_command(cmd)
  if retcode:
    if len(devices) > 1:
      create_drive_raid(data_dir, devices)
    else:
      create_single_drive(data_dir, devices[0])
  else:
    logging.info(
        'Skipping drive creation since path {} already exists'.format(data_dir))


def create_single_drive(data_dir, device):
  """Creates a data drive out of a single device."""
  cmds = []
  cmds.append('sudo mkfs.ext4 -F {}'.format(device))
  cmds.append('sudo mkdir -p {}'.format(data_dir))
  cmds.append('sudo mount {} {}'.format(device, data_dir))
  cmds.append('sudo chmod a+w {}'.format(data_dir))

  utils.run_commands(cmds)
  logging.info('Created and mounted device {} at {}'.format(device, data_dir))


def create_drive_raid(data_dir, list_of_devices):
  """Creates a raid zero array of nvme drives."""
  cmds = []
  # Passing 'yes' because GCE nvme drive are sometimes in an odd state and
  # think they are in another raid. mdadm does not have -y option.
  # Or the kokoro images were left dirty? and that is where the info
  # comes from.
  cmds.append('yes | sudo mdadm --create /dev/md0 --level=0 '
              '--raid-devices={} {}'.format(
                  len(list_of_devices), ' '.join(list_of_devices)))
  cmds.append('sudo mkfs.ext4 -F /dev/md0')
  cmds.append('sudo mkdir -p {}'.format(data_dir))
  cmds.append('sudo mount /dev/md0 {}'.format(data_dir))
  cmds.append('sudo chmod a+w {}'.format(data_dir))

  utils.run_commands(cmds)
  logging.info('Created and mounted RAID array at {}'.format(data_dir))


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
    logging.info('Created RAM disk at {}'.format(data_dir))
  else:
    logging.debug(
        'RAM disk or something else is mounted at {}'.format(data_dir))
