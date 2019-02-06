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
import logging
import os

import perfzero.utils as utils


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dockerfile_path',
      type=str,
      default='docker/Dockerfile',
      help='Path to docker file')
  FLAGS, unparsed = parser.parse_known_args()

  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

  project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
  dockerfile_path = os.path.join(project_dir, FLAGS.dockerfile_path)
  docker_tag = 'temp/tf-gpu'

  cmd = 'docker build --pull -t {} - < {}'.format(docker_tag, dockerfile_path)
  utils.run_commands([cmd])
  logging.info('Built docker image with tag %s', docker_tag)
