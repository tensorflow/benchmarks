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
"""Checkout repository, download data and build docker image."""
from __future__ import print_function

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time

import perfzero.device_utils as device_utils
import perfzero.perfzero_config as perfzero_config
import perfzero.utils as utils


def _temporary_file_name(parent_dir, base_name):
  """Returns a temp name of the form <parent-dir>/<random>/<base-name>."""
  if not os.path.isdir(parent_dir):
    os.makedirs(parent_dir)
  temp_dir = tempfile.mkdtemp(dir=parent_dir)
  return os.path.join(temp_dir, base_name)


def _load_docker_image(FLAGS, workspace_dir, setup_execution_time):
  """Runs docker load --input_image <FLAGS.dockerfile_path>.

  Fetches FLAGS.dockerfile_path to workspace_dir/<temp-dir>/local_docker first.
  Runs docker load --input <path-to-local-docker>.
  Deletes workspace_dir/<temp-dir> after the docker image is loaded.

  Args:
    FLAGS: parser.parse_known_args object.
    workspace_dir: String - The path to use for intermediate artifacts.
    setup_execution_time: Map from string->double containing wall times for
      different operations. This will have insertions describing the docker
      setup time.
  """
  load_docker_start_time = time.time()
  local_docker_image_path = _temporary_file_name(workspace_dir, 'local_docker')
  utils.download_data([{'url': FLAGS.dockerfile_path,
                        'local_path': local_docker_image_path,
                        'decompress': False}])

  setup_execution_time['fetch_docker'] = time.time() - load_docker_start_time

  docker_load_cmd = 'docker load --input {}'.format(local_docker_image_path)
  try:
    utils.run_commands(
        [docker_load_cmd,
         'docker images'  # Print loaded image list.
        ])
    setup_execution_time['load_docker'] = time.time() - load_docker_start_time
  finally:
    logging.info('removing parent dir of local docker image copy %s',
                 local_docker_image_path)
    shutil.rmtree(os.path.dirname(local_docker_image_path))


def _create_docker_image(FLAGS, project_dir, workspace_dir,
                         setup_execution_time):
  """Creates a docker image.

  Args:
    FLAGS: parser.parse_known_args object.
    project_dir: String - The current project path.
    workspace_dir: String - The path to use for intermediate artifacts.
    setup_execution_time: Map from string->double containing wall times for
      different operations. This will have insertions describing the docker
      setup time.
  """
  # Create docker image
  docker_start_time = time.time()
  docker_context = os.path.join(workspace_dir, 'resources')
  # Necessary in case we don't have a local .whl file.
  utils.create_empty_file(docker_context, 'EMPTY')

  # Download TensorFlow pip package from Google Cloud Storage and modify package
  # path accordingly, if applicable
  local_tensorflow_pip_spec = None

  if (FLAGS.tensorflow_pip_spec and
      (FLAGS.tensorflow_pip_spec.startswith('gs://') or
       FLAGS.tensorflow_pip_spec.startswith('file://'))):
    local_pip_filename = os.path.basename(FLAGS.tensorflow_pip_spec)
    local_pip_path = os.path.join(docker_context, local_pip_filename)
    utils.download_data([{'url': FLAGS.tensorflow_pip_spec,
                          'local_path': local_pip_path}])
    # Update path to pip wheel file for the Dockerfile. Note that this path has
    # to be relative to the docker context (absolute path will not work).
    FLAGS.tensorflow_pip_spec = local_pip_filename
    local_tensorflow_pip_spec = local_pip_filename
  else:
    local_tensorflow_pip_spec = 'EMPTY'

  dockerfile_path = FLAGS.dockerfile_path
  if not os.path.exists(dockerfile_path):
    # Fall back to the deprecated approach if the user-specified
    # dockerfile_path does not exist
    dockerfile_path = os.path.join(project_dir, FLAGS.dockerfile_path)
  extra_pip_specs = (FLAGS.extra_pip_specs or '').replace(';', '')
  docker_base_cmd = 'docker build --no-cache --pull'
  # FLAGS.extra_docker_build_args will be a list of strings (e.g. ['a', 'b=c']).
  # We treat the strings directly as build-args: --build-arg a --build-arg b=c
  # Empty strings are ignored.
  extra_docker_build_args = ' '.join([
      '--build-arg %s' % arg for arg in FLAGS.extra_docker_build_args if arg])
  cmd = '{docker_base_cmd} -t {docker_tag}{tf_pip}{local_tf_pip}{extra_pip}{extra_docker_build_args} {suffix}'.format(
      docker_base_cmd=docker_base_cmd,
      docker_tag=FLAGS.docker_tag,
      tf_pip=(
          ' --build-arg tensorflow_pip_spec={}'.format(
              FLAGS.tensorflow_pip_spec) if FLAGS.tensorflow_pip_spec else ''),
      # local_tensorflow_pip_spec is either string 'EMPTY' or basename of
      # local .whl file.
      local_tf_pip=' --build-arg local_tensorflow_pip_spec={}'.format(
          local_tensorflow_pip_spec),
      extra_pip=' --build-arg extra_pip_specs=\'{}\''.format(extra_pip_specs),
      extra_docker_build_args=' ' + extra_docker_build_args,
      suffix=(
          '-f {} {}'.format(dockerfile_path, docker_context)
          if docker_context else '- < {}'.format(dockerfile_path))
  )

  utils.run_commands([cmd])
  logging.info('Built docker image with tag %s', FLAGS.docker_tag)
  setup_execution_time['build_docker'] = time.time() - docker_start_time


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  perfzero_config.add_setup_parser_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()

  logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                      level=logging.DEBUG)
  if unparsed:
    logging.error('Arguments %s are not recognized', unparsed)
    sys.exit(1)

  setup_execution_time = {}
  project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
  workspace_dir = os.path.join(project_dir, FLAGS.workspace)
  site_package_dir = os.path.join(workspace_dir, 'site-packages')
  utils.copy_and_rename_dirs(FLAGS.site_package_downloads,
                             site_package_dir)

  activate_gcloud = False
  if FLAGS.dockerfile_path and FLAGS.dockerfile_path.startswith('gs://'):
    # We might end up doing gsutil fetch later, so need to call
    # active_gcloud_service().
    activate_gcloud = True

  if FLAGS.tensorflow_pip_spec and FLAGS.tensorflow_pip_spec.startswith('gs://'):
    activate_gcloud = True

  # Download gcloud auth token. Remove this operation in the future when
  # docker in Kokoro can accesss the GCP metadata server
  start_time = time.time()
  utils.active_gcloud_service(FLAGS.gcloud_key_file_url,
                              workspace_dir, download_only=not activate_gcloud)
  setup_execution_time['download_token'] = time.time() - start_time

  # Set up the raid array.
  start_time = time.time()
  device_utils.create_drive_from_devices(FLAGS.root_data_dir,
                                         FLAGS.gce_nvme_raid)
  setup_execution_time['create_drive'] = time.time() - start_time

  if FLAGS.dockerfile_path:
    if FLAGS.dockerfile_path.endswith('.tar.gz'):
      logging.info('Assuming given file %s is a docker image to load',
                   FLAGS.dockerfile_path)
      _load_docker_image(FLAGS, workspace_dir,
                         setup_execution_time)
    else:
      _create_docker_image(FLAGS, project_dir, workspace_dir,
                           setup_execution_time)

  logging.info('Setup time in seconds by operation:\n %s',
               json.dumps(setup_execution_time, indent=2))
