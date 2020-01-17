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
  _log = logging.info if is_from_user else logging.debug
  _log('Executing command: {}'.format(cmd))
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, shell=True)

  exit_code = None
  line = ''
  stdout = ''
  while exit_code is None or line:
    exit_code = p.poll()
    line = p.stdout.readline().decode('utf-8')
    stdout += line
    _log(line)
  if exit_code and is_from_user:
    sys.exit(exit_code)
  elif exit_code:
    raise Exception('Command:\n{}\nfailed with output:\n{}'.format(cmd, stdout))

  return stdout


def get_instance_name(username):
  return INSTANCE_NAME_PREFIX + username


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


def _ssh_prefix(project, zone, internal_ip, key_file):
  if internal_ip:
    ssh_prefix = 'gcloud beta compute ssh --internal-ip'
  else:
    ssh_prefix = 'gcloud compute ssh'
  if key_file:
    ssh_prefix = '{} --ssh-key-file={}'.format(ssh_prefix, key_file)
  return '{} --project={} --zone={}'.format(ssh_prefix, project, zone)


def create(username, project, zone, machine_type, accelerator_count,
           accelerator_type, image, nvme_count, ssh_internal_ip, ssh_key_file,
           cpu_min_platform=None, boot_ssd_size=None):
  """Create gcloud computing instance.

  Args:
    username: the username of the current user
    project: project name
    zone: zone of the GCP computing instance
    machine_type: the machine type used for the instance
    accelerator_count: the number of pieces of the accelerator to attach to
                       the instance
    accelerator_type: the specific type of accelerator to attach to the instance
    image: the name of the image that the disk will be initialized with
    nvme_count: the number of NVME local SSD devices to attach to the instance
    ssh_internal_ip: internal ip to use for ssh.
    ssh_key_file: ssh key file to use to connect to instance.
    cpu_min_platform: minimum CPU platform to use, if None use default.
    boot_ssd_size: If set boot disk is changed to SSD and this size(GB) is used.
  """
  instance_name = get_instance_name(username)
  machine_type = get_machine_type(machine_type, accelerator_count)
  logging.debug('Creating gcloud computing instance %s', instance_name)

  cmd = '''gcloud compute instances create {} \
--image={} \
--project={} \
--zone={} \
--machine-type={} \
--maintenance-policy=TERMINATE \
'''.format(instance_name, image, project, zone, machine_type)

  if boot_ssd_size:
    cmd += '--boot-disk-size={}GB --boot-disk-type=pd-ssd '.format(
        boot_ssd_size)

  if accelerator_count > 0:
    cmd += '--accelerator=count={},type={} '.format(
        accelerator_count, accelerator_type)

  if cpu_min_platform:
    cmd += '--min-cpu-platform="{}" '.format(cpu_min_platform)

  for _ in range(nvme_count):
    cmd += '--local-ssd=interface=NVME '

  run_command(cmd, is_from_user=True)
  logging.info('Successfully created gcloud computing instance %s '
               'with %s accelerator.\n', instance_name, accelerator_count)

  ssh_prefix = _ssh_prefix(project, zone, ssh_internal_ip, ssh_key_file)
  # Wait until we can ssh to the newly created computing instance
  cmd = '{} --strict-host-key-checking=no --command="exit" {}'.format(
      ssh_prefix, instance_name)
  ssh_remaining_retries = 12
  ssh_error = None
  while ssh_remaining_retries > 0:
    ssh_remaining_retries -= 1
    try:
      run_command(cmd, is_from_user=False)
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
  else:
    cmd = '{} --command="git clone {}" {}'.format(
        ssh_prefix, 'https://github.com/tensorflow/benchmarks.git',
        instance_name)
    run_command(cmd, is_from_user=True)
    logging.info('Successfully checked-out PerfZero code on the '
                 'computing instance\n')

    cmd = '{} --command="sudo usermod -a -G docker $USER" {}'.format(
        ssh_prefix, instance_name)
    run_command(cmd, is_from_user=True)
    logging.info('Successfully added user to the docker group\n')

  cmd = '{} {} -- -L 6006:127.0.0.1:6006'.format(ssh_prefix, instance_name)
  logging.info('Run the command below to ssh to the instance together with '
               'port forwarding for tensorboard:\n%s\n', cmd)


def status(username, project, zone, ssh_internal_ip, ssh_key_file):
  """Query the status of the computing instance.

  Args:
    username: the username of the current user.
    project: project name.
    zone: zone of the GCP computing instance.
    ssh_internal_ip: internal ip of the instance.
    ssh_key_file: SSH key file to use to connect to the instance.
  """
  instance_name = get_instance_name(username)
  logging.debug('Querying status of gcloud computing instance %s of '
                'project %s in zone %s', instance_name, project, zone)

  cmd = 'gcloud compute instances list --filter="name={} AND zone:{}" --project {}'.format(  # pylint: disable=line-too-long
      instance_name, zone, project)
  stdout = run_command(cmd, is_from_user=True)

  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance with name %s.\n',
               num_instances, instance_name)

  if num_instances == 1:
    cmd = '{} {} -- -L 6006:127.0.0.1:6006'.format(
        _ssh_prefix(project, zone, ssh_internal_ip, ssh_key_file),
        instance_name)
    logging.info('Run the command below to ssh to the instance together with '
                 'port forwarding for tensorboard:\n%s\n', cmd)


def list_all(project):
  logging.debug('Finding all gcloud computing instance of project %s created '
                'for PerfZero test', project)
  cmd = 'gcloud compute instances list --filter="name ~ {}" --project={}'.format(  # pylint: disable=line-too-long
      INSTANCE_NAME_PREFIX, project)
  stdout = run_command(cmd, is_from_user=True)
  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance of project %s created '
               'for PerfZero test', num_instances, project)


def start(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Starting gcloud computing instance %s of project %s '
                'in zone %s', instance_name, project, zone)

  cmd = 'gcloud compute instances start {} --project={} --zone={}'.format(
      instance_name, project, zone)
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully started gcloud computing instance %s of '
                'project %s in zone %s', instance_name, project, zone)


def stop(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Stopping gcloud computing instance %s of project %s in '
                'zone %s', instance_name, project, zone)

  cmd = 'gcloud compute instances stop {} --project={} --zone={}'.format(
      instance_name, project, zone)
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully stopped gcloud computing instance %s of '
                'project %s in zone %s', instance_name, project, zone)


def delete(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Deleting gcloud computing instance %s of project %s in '
                'zone %s', instance_name, project, zone)

  cmd = 'echo Y | gcloud compute instances delete {} --project={} --zone={}'.format(  # pylint: disable=line-too-long
      instance_name, project, zone)
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully deleted gcloud computing instance %s of '
                'project %s in zone %s', instance_name, project, zone)


def parse_arguments(argv, command):  # pylint: disable=redefined-outer-name
  """Parse command line arguments and return parsed flags.

  Args:
    argv: command line arguments
    command: the subcommand requested by the user

  Returns:
    parsed flags
  """

  # pylint: disable=redefined-outer-name
  parser = argparse.ArgumentParser(
      usage='cloud_manager.py {} [<args>]'.format(command),
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--debug',
      action='store_true',
      help='If set, use debug level logging. Otherwise, use info level logging'
      )
  parser.add_argument(
      '--project',
      default='google.com:tensorflow-performance',
      type=str,
      help='Google Cloud Platform project name to use for this invocation'
      )

  if command in ['create', 'start', 'stop', 'delete', 'status']:
    parser.add_argument(
        '--username',
        default=getpass.getuser(),
        type=str,
        help='''Username that uniquely identifies the name of computing instance created for PerfZero.
        The default value is your ldap username.
        ''')
    parser.add_argument(
        '--zone',
        default='us-west1-b',
        type=str,
        help='Zone of the instance to create.'
        )
    parser.add_argument(
        '--ssh-internal-ip',
        action='store_true',
        help='If set, use internal IP for ssh with `gcloud beta compute ssh`.'
        )
    parser.add_argument(
        '--ssh-key-file',
        default=None,
        type=str,
        help='The ssh key to use with with `gcloud (beta) compute ssh`.'
        )

  if command == 'create':
    parser.add_argument(
        '--accelerator_count',
        default=1,
        type=int,
        help='The number of pieces of the accelerator to attach to the instance'
        )
    parser.add_argument(
        '--accelerator_type',
        default='nvidia-tesla-v100',
        type=str,
        help='''The specific type (e.g. nvidia-tesla-v100 for nVidia Tesla V100) of
        accelerator to attach to the instance. Use 'gcloud compute accelerator-types list --project=${project_name}' to
        learn about all available accelerator types.
        ''')
    parser.add_argument(
        '--cpu_min_platform',
        default=None,
        type=str,
        help='''Minimum cpu platform, only needed for CPU only instances.''')
    parser.add_argument(
        '--machine_type',
        default=None,
        type=str,
        help='''The machine type used for the instance. To get a list of available machine
        types, run 'gcloud compute machine-types list --project=${project_name}'
        ''')
    parser.add_argument(
        '--image',
        default='tf-ubuntu-1604-20180927-410',
        type=str,
        help='''Specifies the name of the image that the disk will be initialized with.
        A new disk will be created based on the given image. To view a list of
        public images and projects, run 'gcloud compute images list --project=${project_name}'. It is best
        practice to use image when a specific version of an image is needed.
        ''')
    parser.add_argument(
        '--nvme_count',
        default=0,
        type=int,
        help='''Specifies the number of NVME local SSD devices to attach to the instance.
        '''
        )
    parser.add_argument(
        '--boot_ssd_size',
        default=None,
        type=int,
        help='''Specifies the size (GB) of the boot disk or size is the image
        size. Setting this also changes boot disk to Persistent SSD.
        '''
        )

  flags, unparsed = parser.parse_known_args(argv)  # pylint: disable=redefined-outer-name
  if unparsed:
    logging.error('Arguments %s are not recognized', unparsed)
    sys.exit(1)

  level = logging.DEBUG if flags.debug else logging.INFO
  logging.basicConfig(format='%(message)s', level=level)

  return flags
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      usage='''cloud_manager.py <command> [<args>]

The supported commands are:
  create:  Create a computing instance in gcloud that is unique to the specified username, which is your ldap by default
  start:   Start the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  stop:    Stop the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  delete:  Delete the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  status:  Query the status and information of the computing instance in gcloud that is unique to the specified username, which is your ldap by default
  list_all:    Query the status of all computing instances that are created by this script.'''
  )
  parser.add_argument(
      'command',
      type=str
      )

  flags = parser.parse_args(sys.argv[1:2])
  command = flags.command
  if not hasattr(sys.modules[__name__], command):
    print('Error: The command <{}> is not recognized\n'.format(command))
    parser.print_help()
    sys.exit(1)

  flags = parse_arguments(sys.argv[2:], command)

  if command == 'create':
    create(flags.username, flags.project, flags.zone, flags.machine_type,
           flags.accelerator_count, flags.accelerator_type, flags.image,
           flags.nvme_count, flags.ssh_internal_ip, flags.ssh_key_file,
           cpu_min_platform=flags.cpu_min_platform,
           boot_ssd_size=flags.boot_ssd_size)
  elif command == 'start':
    start(flags.username, flags.project, flags.zone)
  elif command == 'stop':
    stop(flags.username, flags.project, flags.zone)
  elif command == 'delete':
    delete(flags.username, flags.project, flags.zone)
  elif command == 'status':
    status(flags.username, flags.project, flags.zone, flags.ssh_internal_ip,
           flags.ssh_key_file)
  elif command == 'list_all':
    list_all(flags.project)


