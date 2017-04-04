# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Builds docker images and runs benchmarks from benchmark_configs.yml file.

This script should only be run from opensource repository.
"""
import argparse
from datetime import datetime
import logging
import os
from string import maketrans
import subprocess

import docker
import k8s_tensorflow_lib
import kubectl_util
import yaml


_DOCKER_IMAGE_PATTERN = 'gcr.io/tensorflow-testing/benchmarks/%s'
_OUTPUT_FILE_ENV_VAR = 'TF_DIST_BENCHMARK_RESULTS_FILE'
_TEST_NAME_ENV_VAR = 'TF_DIST_BENCHMARK_NAME'
_PORT = 5000


def _ConvertToValidName(name):
  """Converts to name that we can use as a kubernetes job prefix.

  Args:
    name: benchmark name.

  Returns:
    Benchmark name that can be used as a kubernetes job prefix.
  """
  return name.translate(maketrans('/:_', '---'))


def _RunBenchmark(name, yaml_file):
  """Runs a single distributed benchmark.

  Args:
    name: name of the benchmark to run.
    yaml_file: path to kubernetes config file.
  """
  kubectl_util.DeletePods(name, yaml_file)
  kubectl_util.CreatePods(name, yaml_file)
  kubectl_util.WaitForCompletion(name)
  kubectl_util.DeletePods(name, yaml_file)


def _BuildAndPushDockerImage(
    docker_client, docker_file, name, tag, push_to_gcloud=False):
  """Builds a docker image and optionally pushes it to gcloud.

  Args:
    docker_client: docker.Client object.
    docker_file: Dockerfile path.
    name: name of the benchmark to build a docker image for.
    tag: tag for docker image.
    push_to_gcloud: whether to push the image to google cloud.

  Returns:
    Docker image identifier.
  """
  local_docker_image_with_tag = '%s:%s' % (name, tag)
  remote_docker_image = _DOCKER_IMAGE_PATTERN % name
  remote_docker_image_with_tag = '%s:%s' % (remote_docker_image, tag)
  if FLAGS.docker_context_dir:
    docker_context = os.path.join(
        os.path.dirname(__file__), FLAGS.docker_context_dir)
    docker_file_name = docker_file
  else:
    docker_context = os.path.dirname(docker_file)
    docker_file_name = os.path.basename(docker_file)

  built_image = docker_client.images.build(
      path=docker_context, dockerfile=docker_file_name,
      tag=local_docker_image_with_tag,
      pull=True)
  built_image.tag(remote_docker_image, tag=tag)
  if push_to_gcloud:
    subprocess.check_call(
        ['gcloud', 'docker', '--', 'push', remote_docker_image_with_tag])
  return remote_docker_image_with_tag


def main():
  config_file = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), 'benchmark_configs.yml')
  config_text = open(config_file, 'r').read()
  configs = yaml.load(config_text)

  docker_client = docker.from_env()
  time_tag = datetime.now().strftime('%d_%m_%Y_%H_%M')
  # Create directories to store kubernetes yaml configs in.
  if not os.path.isdir(FLAGS.config_output_file_dir):
    os.makedirs(FLAGS.config_output_file_dir)
  # Keeps track of already built docker images in case multiple benchmarks
  # use the same docker image.
  benchmark_name_to_docker_image = {}

  # TODO(annarev): run benchmarks in parallel instead of sequentially.
  for config in configs:
    name = _ConvertToValidName(str(config['benchmark_name']))
    if name in benchmark_name_to_docker_image:
      docker_image = benchmark_name_to_docker_image[name]
    else:
      docker_image = _BuildAndPushDockerImage(
          docker_client, config['docker_file'], name, time_tag,
          FLAGS.store_docker_image_in_gcloud)
      benchmark_name_to_docker_image[name] = docker_image
    env_vars = {
        _OUTPUT_FILE_ENV_VAR: os.path.join(
            FLAGS.benchmark_results_dir, name + '.json'),
        _TEST_NAME_ENV_VAR: name
    }
    env_vars.update(config.get('env_vars', {}))
    args = config.get('args', {})
    kubernetes_config = k8s_tensorflow_lib.GenerateConfig(
        config['worker_count'],
        config['ps_count'],
        _PORT,
        request_load_balancer=False,
        docker_image=docker_image,
        name_prefix=name,
        additional_args=args,
        env_vars=env_vars,
        use_shared_volume=False,
        use_cluster_spec=False)

    kubernetes_config_path = os.path.join(
        FLAGS.config_output_file_dir, name + '.yaml')
    with open(kubernetes_config_path, 'w') as output_config_file:
      output_config_file.write(kubernetes_config)

    _RunBenchmark(name, kubernetes_config_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--config_output_file_dir', type=str, default=None, required=True,
      help='Directory to write generated kubernetes configs to.')
  parser.add_argument(
      '--benchmark_results_dir', type=str, default=None, required=True,
      help='Directory to store benchmark results at.')
  parser.add_argument(
      '--docker_context_dir', type=str, default='',
      help='Directory to use as a docker context. By default, docker context '
           'will be set to the directory containing a docker file.')
  parser.add_argument(
      '--store_docker_image_in_gcloud', type='bool', nargs='?', const=True,
      default=False, help='Push docker images to google cloud.')
  FLAGS, _ = parser.parse_known_args()
  logging.basicConfig(level=logging.DEBUG)
  main()
