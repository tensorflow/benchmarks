#!/usr/bin/python
#
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
"""Helper script to create, query and stop machine in GCP."""

from __future__ import print_function

import argparse
import getpass
import logging
import subprocess
import sys
import time


INSTANCE_NAME_PREFIX = 'perfzero-dev-'


def run_command(cmd, is_from_user=False):
  """Runs list of command and throw error if return code is non-zero.

  Args:
    cmd: Command to execute
    is_from_user: If true, log the command and the command output in INFO level.
                  Otherwise, log these in the DEBUG level.

  Returns:
    a string representing the command output

  Raises:
    Exception: raised when the command execution has non-zero exit code
  """
  if is_from_user:
    logging.info('Executing command: %s\n', cmd)
  else:
    logging.debug('Executing command: %s\n', cmd)

  stdout = ''
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, shell=True)
  exit_code = None
  while exit_code is None:
    exit_code = p.poll()
    line = p.stdout.readline().decode('utf-8').strip()
    if not line:
      continue

    if is_from_user:
      logging.info(line)
    else:
      logging.debug(line)
    stdout = stdout + line + '\n'

  if exit_code and is_from_user:
    sys.exit(exit_code)
  elif exit_code:
    raise Exception('Command:\n{}\nfailed with output:\n{}'.format(cmd, stdout))

  return stdout


def get_machine_type(machine_type, accelerator_count):
  """Get machine type for the instance.

  - Use the user-specified machine_type if it is not None
  - Otherwise, use the standard type with cpu_count = 8 x accelerator_count
    if user-specified accelerator_count > 0
  - Otherwise, use the standard type with 8 cpu

  Args:
    machine_type: machine_type specified by the user
    accelerator_count: accelerator count

  Returns:
    the machine type used for the instance
  """
  if machine_type:
    return machine_type
  cpu_count = max(accelerator_count, 1) * 8
  return 'n1-standard-{}'.format(cpu_count)


def _ssh_prefix(instance_name, project, zone):
  return 'gcloud compute ssh {} --project={} --zone={}'.format(
      instance_name, project, zone)


def _ssh_cmd(instance_name, project, zone, remote_cmd,
             strict_host_key_checking=True):
  ssh_cmd = '{} --command="{}"'.format(
      _ssh_prefix(instance_name, project, zone), remote_cmd)
  if not strict_host_key_checking:
    ssh_cmd += ' --strict-host-key-checking=no'
  return ssh_cmd


def _setup_after_create(instance_name, project, zone):
  logging.info('Post-creation setup for instance {}'.format(instance_name))
  ssh_remaining_retries = 12
  ssh_error = None
  while ssh_remaining_retries > 0:
    ssh_remaining_retries -= 1
    try:
      run_command(_ssh_cmd(instance_name, project, zone, 'exit', False),
                  is_from_user=False)
      ssh_error = None
    except Exception as error:  # pylint: disable=broad-except
      ssh_error = error
      if ssh_remaining_retries:
        logging.info('Cannot ssh to the computing instance. '
                     'Try again after 5 seconds')
        time.sleep(5)
      else:
        logging.error('Cannot ssh to the computing instance after '
                      '60 seconds due to error:\n%s', str(ssh_error))

  if ssh_error:
    logging.info('Run the commands below manually after ssh into the computing '
                 'instance:\n'
                 'git clone https://github.com/tensorflow/benchmarks.git\n'
                 'sudo usermod -a -G docker $USER\n')
    return

  remote_cmd = 'git clone https://github.com/tensorflow/benchmarks.git'
  run_command(_ssh_cmd(instance_name, project, zone, remote_cmd),
              is_from_user=True)
  logging.info('Successfully checked-out PerfZero code on the '
               'computing instance\n')

  remote_cmd = 'sudo usermod -a -G docker $USER'
  run_command(_ssh_cmd(instance_name, project, zone, remote_cmd),
              is_from_user=True)
  logging.info('Successfully added user to the docker group\n')

  port_forwarding_cmd = (_ssh_prefix(instance_name, project, zone) +
                         ' -- -L 6006:127.0.0.1:6006')
  logging.info('Run the command below to ssh to the instance together with '
               'port forwarding for tensorboard:\n%s\n', port_forwarding_cmd)


def _instance_names(instance_name, num_instances):
  if num_instances == 1:
    return [instance_name]
  else:
    return ['%s-%d' % (instance_name, i) for i in range(num_instances)]


def _instances_cmd(command, instance_names, project, zone):
  """Return a string 'gcloud compute instances' command.

  Args:
    command: command that goes with 'gcloud compute instances'.
    instance_names: list of instance names.
    project: project name
    zone: GCP zone

  Returns:
    string command
  """
  return 'gcloud compute instances {} {} --project={} --zone={}'.format(
      command, ' '.join(instance_names), project, zone)


def create(instance_name, project, zone, machine_type, accelerator_count,
           accelerator_type, image, nvme_count, num_instances):
  """Create gcloud computing instance.

  Args:
    instance_name: instance_name
    project: project name
    zone: zone of the GCP computing instance
    machine_type: the machine type used for the instance
    accelerator_count: the number of pieces of the accelerator to attach to
                       the instance
    accelerator_type: the specific type of accelerator to attach to the instance
    image: the name of the image that the disk will be initialized with
    nvme_count: the number of NVME local SSD devices to attach to the instance
  """
  instance_names = _instance_names(instance_name, num_instances)
  logging.debug('Creating gcloud computing instances {}'.format(instance_names))
  machine_type = get_machine_type(machine_type, accelerator_count)
  cmd = (_instances_cmd('create', instance_names, project, zone) +
        ' --image={} --machine-type={} --maintenance-policy=TERMINATE '.format(
            image, machine_type))
  if accelerator_count > 0:
    cmd += '--accelerator=count={},type={} '.format(
        accelerator_count, accelerator_type)
  for _ in range(nvme_count):
    cmd += '--local-ssd=interface=NVME '

  run_command(cmd, is_from_user=True)
  logging.info('Successfully created gcloud computing instances {} '
               'with {} accelerators.\n'.format(
                   instance_names, accelerator_count))

  # Wait until we can ssh to the newly created computing instances
  for instance_name in instance_names:
    _setup_after_create(instance_name, project, zone)


def status(instance_name, project, zone):
  """Query the status of the computing instance.

  Args:
    instance_name: instance name
    project: project name
    zone: zone of the GCP computing instance
  """
  logging.debug('Querying status of gcloud computing instance %s of '
                'project %s in zone %s', instance_name, project, zone)

  cmd = 'gcloud compute instances list --filter="name={} AND zone:{}" --project {}'.format(  # pylint: disable=line-too-long
      instance_name, zone, project)
  stdout = run_command(cmd, is_from_user=True)

  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance with name %s.\n',
               num_instances, instance_name)

  if num_instances == 1:
    port_forwarding_cmd = (_ssh_prefix(instance_name, project, zone) + 
                           ' -- -L 6006:127.0.0.1:6006')
    logging.info('Run the command below to ssh to the instance together with '
                 'port forwarding for tensorboard:\n%s\n', port_forwarding_cmd)


def list_all(project):
  logging.debug('Finding all gcloud computing instance of project %s created '
                'for PerfZero test', project)
  cmd = 'gcloud compute instances list --filter="name ~ {}" --project={}'.format(  # pylint: disable=line-too-long
      INSTANCE_NAME_PREFIX, project)
  stdout = run_command(cmd, is_from_user=True)
  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance of project %s created '
               'for PerfZero test', num_instances, project)


def start(instance_name, project, zone, num_instances):
  instance_names = _instance_names(instance_name, num_instances)
  logging.debug('Starting gcloud computing instance {} of project {} '
                'in zone {}'.format(instance_names, project, zone))
  cmd = _instances_cmd('start', instance_names, project, zone)
  run_command(cmd , is_from_user=True)
  logging.info('Successfully started gcloud computing instances {} of '
               'project {} in zone {}'.format(instance_names, project, zone))


def stop(instance_name, project, zone, num_instances):
  instance_names = _instance_names(instance_name, num_instances)
  logging.debug('Stopping gcloud computing instance {} of project {} in '
                'zone {}'.format(instance_name, project, zone))
  cmd = _instances_cmd('stop', instance_names, project, zone)
  run_command(cmd, is_from_user=True)
  logging.info('Successfully stopped gcloud computing instances {} of '
               'project {} in zone {}'.format(instance_names, project, zone))


def delete(instance_name, project, zone, num_instances):
  instance_names = _instance_names(instance_name, num_instances)
  logging.debug('Deleting gcloud computing instance {} of project {} in '
                'zone {}'.format(instance_name, project, zone))
  cmd = 'echo Y | ' + _instances_cmd('delete', instance_names, project, zone)
  run_command(cmd, is_from_user=True)
  logging.info('Successfully deleted gcloud computing instances {} of '
               'project {} in zone {}'.format(instance_names, project, zone))


def parse_arguments():  # pylint: disable=redefined-outer-name
  """Parse command line arguments and return parsed flags.

  Returns:
    parsed flags
  """
  parser = argparse.ArgumentParser(
      epilog='''The supported commands are:
  create:  Create a computing instance in gcloud that is unique to the specified username, which is your ldap by default
  start:   Start the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  stop:    Stop the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  delete:  Delete the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  status:  Query the status and information of the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  list_all:    Query the status of all computing instances that are created by this script.''',
  formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      'command',
      type=str,
      help='Accepted values: create, start, stop, delete, status, list_all.')
  parser.add_argument(
      '--debug',
      action='store_true',
      help='If set, use debug level logging. Otherwise, use info level logging')
  parser.add_argument(
      '--project',
      default=None,
      type=str,
      help='Google Cloud Platform project name to use for this invocation',
      required=True)
  parser.add_argument(
      '--instance_name',
      default=INSTANCE_NAME_PREFIX + getpass.getuser(),
      type=str,
      help='''Name of instance for PerfZero.  If num_instances is more than 1,
      then this is the prefix for each instance's name.  Default:
      {}<username>.'''.format(INSTANCE_NAME_PREFIX))
  parser.add_argument(
      '--num_instances',
      default=1,
      type=int,
      help='Number of instance for bulk creation/deletion.  Default 1.')
  parser.add_argument(
      '--zone',
      default='us-west1-b',
      type=str,
      help='Zone of the instance to create.')
  parser.add_argument(
      '--accelerator_count',
      default=1,
      type=int,
      help='The number of pieces of the accelerator to attach to the instance')
  parser.add_argument(
      '--accelerator_type',
      default='nvidia-tesla-v100',
      type=str,
      help='''The specific type (e.g. nvidia-tesla-v100 for nVidia Tesla V100) of
      accelerator to attach to the instance. Use 'gcloud compute accelerator-types list --project=${project_name}' to
      learn about all available accelerator types.''')
  parser.add_argument(
      '--machine_type',
      default='n1-highmem-64',
      type=str,
      help='''The machine type used for the instance. To get a list of available machine
      types, run 'gcloud compute machine-types list --project=${project_name}'.''')
  parser.add_argument(
      '--image',
      default=None,
      type=str,
      help='''Specifies the name of the image that the disk will be initialized with.
      A new disk will be created based on the given image. To view a list of
      public images and projects, run 'gcloud compute images list --project=${project_name}'. It is best
      practice to use image when a specific version of an image is needed.''')
  parser.add_argument(
      '--nvme_count',
      default=0,
      type=int,
      help='Specifies the number of NVME local SSD devices to attach to the instance.')

  flags = parser.parse_args()

  if not hasattr(sys.modules[__name__], flags.command):
    print('Error: The command <{}> is not recognized\n'.format(flags.command))
    parser.print_help()
    sys.exit(1)

  if flags.command == 'create':
    if not flags.machine_type or not flags.image:
      raise ValueError('Need to specify machine_type and image for creating '
                       'instances')

  level = logging.DEBUG if flags.debug else logging.INFO
  logging.basicConfig(format='%(message)s', level=level)

  return flags


if __name__ == '__main__':
  flags = parse_arguments()
  command = flags.command
  if command == 'create':
    create(flags.instance_name, flags.project, flags.zone, flags.machine_type,
           flags.accelerator_count, flags.accelerator_type, flags.image,
           flags.nvme_count, flags.num_instances)
  elif command == 'start':
    start(flags.instance_name, flags.project, flags.zone, flags.num_instances)
  elif command == 'stop':
    stop(flags.instance_name, flags.project, flags.zone, flags.num_instances)
  elif command == 'delete':
    delete(flags.instance_name, flags.project, flags.zone, flags.num_instances)
  elif command == 'status':
    status(flags.instance_name, flags.project, flags.zone)
  elif command == 'list_all':
    list_all(flags.project)
