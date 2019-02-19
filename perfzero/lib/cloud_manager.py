#!/usr/bin/python

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


def get_instance_name(username):
  return INSTANCE_NAME_PREFIX + username


def get_machine_type(accelerator_count):
  """Get machine type string based on the accelerator count.

  Args:
    accelerator_count: accelerator count

  Returns:
    a string recognized by the gcloud utility as the machine type
  """
  if accelerator_count == 0:
    return 'n1-standard-2'
  elif accelerator_count == 1:
    return 'n1-standard-4'
  elif accelerator_count == 2:
    return 'n1-standard-8'
  elif accelerator_count <= 4:
    return 'n1-standard-16'
  elif accelerator_count <= 8:
    return 'n1-standard-32'
  elif accelerator_count <= 16:
    return 'n1-standard-64'


def create(username, project, zone, accelerator_count):
  """Create gcloud computing instance.

  Args:
    username: the username of the current user
    project: project name
    zone: zone of the GCP computing instance
    accelerator_count: accelerator count
  """
  instance_name = get_instance_name(username)
  machine_type = get_machine_type(accelerator_count)
  logging.debug('Creating gcloud computing instance %s', instance_name)

  cmd = '''gcloud compute instances create {} \
--image tf-ubuntu-1604-20180927-410 \
--project {} \
--zone {} \
--machine-type {} \
--maintenance-policy TERMINATE \
'''.format(instance_name, project, zone, machine_type)

  if accelerator_count > 0:
    cmd += '--accelerator count={},type=nvidia-tesla-v100'.format(accelerator_count)  # pylint: disable=line-too-long

  run_command(cmd, is_from_user=True)
  logging.info('Successfully created gcloud computing instance %s with %s accelerator.\n',  # pylint: disable=line-too-long
               instance_name, accelerator_count)

  cmd = 'gcloud compute ssh {} --project={} --zone={} --command="git clone {}"'.format(
      instance_name, project, zone, 'https://github.com/tensorflow/benchmarks.git')  # pylint: disable=line-too-long
  logging.info('Run the command below to checkout PerfZero code on the computing instance:\n%s\n', cmd)  # pylint: disable=line-too-long

  cmd = 'gcloud compute ssh {} --project={} --zone={} -- -L 6006:127.0.0.1:6006'.format(instance_name, project, zone)  # pylint: disable=line-too-long
  logging.info('Run the command below to ssh to the instance together with port forwarding for tensorboard:\n%s\n', cmd)  # pylint: disable=line-too-long


def status(username, project, zone):
  """Query the status of the computing instance.

  Args:
    username: the username of the current user
    project: project name
    zone: zone of the GCP computing instance
  """
  instance_name = get_instance_name(username)
  logging.debug('Querying status of gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long

  cmd = 'gcloud compute instances list --filter="name={} AND zone:{}" --project {}'.format(instance_name, zone, project)  # pylint: disable=line-too-long
  stdout = run_command(cmd, is_from_user=True)

  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance with name %s.\n', num_instances, instance_name)  # pylint: disable=line-too-long

  if num_instances == 1:
    cmd = 'gcloud compute ssh {} --project={} --zone={} -- -L 6006:127.0.0.1:6006'.format(instance_name, project, zone)  # pylint: disable=line-too-long
    logging.info('Run the command below to ssh to the instance together with port forwarding for tensorboard:\n%s\n', cmd)  # pylint: disable=line-too-long


def list_all(project):
  logging.debug('Finding all gcloud computing instance of project %s created for PerfZero test', project)  # pylint: disable=line-too-long
  cmd = 'gcloud compute instances list --filter="name ~ {}" --project {}'.format(INSTANCE_NAME_PREFIX, project)  # pylint: disable=line-too-long
  stdout = run_command(cmd, is_from_user=True)
  num_instances = len(stdout.splitlines()) - 1
  logging.info('\nFound %s gcloud computing instance of project %s created for PerfZero test', num_instances, project)  # pylint: disable=line-too-long


def start(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Starting gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long

  cmd = 'gcloud compute instances start {} --project {} --zone {}'.format(
      instance_name, project, zone)
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully started gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long


def stop(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Stopping gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long

  cmd = 'gcloud compute instances stop {} --project {} --zone {}'.format(instance_name, project, zone)  # pylint: disable=line-too-long
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully stopped gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long


def delete(username, project, zone):
  instance_name = get_instance_name(username)
  logging.debug('Deleting gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long

  cmd = 'echo Y | gcloud compute instances delete {} --project {} --zone {}'.format(instance_name, project, zone)  # pylint: disable=line-too-long
  run_command(cmd, is_from_user=True)
  logging.debug('\nSuccessfully deleted gcloud computing instance %s of project %s in zone %s', instance_name, project, zone)  # pylint: disable=line-too-long


def parse_arguments(argv, command):  # pylint: disable=redefined-outer-name
  """Parse command line arguments and return parsed flags.

  Args:
    argv: command line arguments
    command: the subcommand requested by the user

  Returns:
    parsed flags
  """

  # pylint: disable=redefined-outer-name
  parser = argparse.ArgumentParser(usage='cloud_manager.py {} [<args>]'.format(command),  # pylint: disable=line-too-long
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # pylint: disable=line-too-long
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
        help='Zone of the instances to manage'
        )

  if command == 'create':
    parser.add_argument(
        '--accelerator_count',
        default=0,
        type=int,
        help='The number of pieces of the accelerator to attach to the instances'
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
  list:    Query the status of all computing instances that are created by this script.'''
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
    create(flags.username, flags.project, flags.zone, flags.accelerator_count)
  elif command == 'start':
    start(flags.username, flags.project, flags.zone)
  elif command == 'stop':
    stop(flags.username, flags.project, flags.zone)
  elif command == 'delete':
    delete(flags.username, flags.project, flags.zone)
  elif command == 'status':
    status(flags.username, flags.project, flags.zone)
  elif command == 'list_all':
    list_all(flags.project)


