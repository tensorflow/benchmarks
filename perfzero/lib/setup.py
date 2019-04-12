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
import sys
import time

import perfzero.device_utils as device_utils
import perfzero.perfzero_config as perfzero_config
import perfzero.utils as utils


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

  # Download gcloud auth token. Remove this operation in the future when
  # docker in Kokoro can accesss the GCP metadata server
  start_time = time.time()
  utils.active_gcloud_service(FLAGS.gcloud_key_file_url,
                              workspace_dir, download_only=True)
  setup_execution_time['download_token'] = time.time() - start_time

  # Set up the raid array.
  start_time = time.time()
  device_utils.create_drive_from_devices(FLAGS.root_data_dir,
                                         FLAGS.gce_nvme_raid)
  setup_execution_time['create_drive'] = time.time() - start_time

  # Create docker image
  start_time = time.time()
  dockerfile_path = FLAGS.dockerfile_path
  if not os.path.exists(dockerfile_path):
    # Fall back to the deprecated approach if the user-specified
    # dockerfile_path does not exist
    dockerfile_path = os.path.join(project_dir, FLAGS.dockerfile_path)
  docker_tag = 'perfzero/tensorflow'
  if FLAGS.tensorflow_pip_spec:
    cmd = 'docker build --no-cache --pull -t {} --build-arg tensorflow_pip_spec={} - < {}'.format(  # pylint: disable=line-too-long
        docker_tag, FLAGS.tensorflow_pip_spec, dockerfile_path)
  else:
    cmd = 'docker build --no-cache --pull -t {} - < {}'.format(docker_tag, dockerfile_path)  # pylint: disable=line-too-long

  utils.run_commands([cmd])
  logging.info('Built docker image with tag %s', docker_tag)
  setup_execution_time['build_docker'] = time.time() - start_time

  logging.info('Setup time in seconds by operation:\n %s',
               json.dumps(setup_execution_time, indent=2))
