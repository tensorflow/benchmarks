"""Generates Kubernetes config yml file from benchmark_configs.yml file.

This script should only be run from opensource repository.
"""
import argparse
import logging
import os
from string import maketrans

import k8s_tensorflow_lib
import yaml


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


def _GetGpuVolumeMounts(flags):
  """Get volume specs to add to Kubernetes config.

  Args:
    flags: flags

  Returns:
    Volume specs in the format: volume_name: (hostPath, podPath).
  """
  volume_specs = {}

  if flags.nvidia_lib_dir:
    volume_specs['nvidia-libraries'] = (flags.nvidia_lib_dir, '/usr/lib/nvidia')

  if flags.cuda_lib_dir:
    cuda_library_files = ['libcuda.so', 'libcuda.so.1', 'libcudart.so']
    for cuda_library_file in cuda_library_files:
      lib_name = cuda_library_file.split('.')[0]
      volume_specs['cuda-libraries-%s' % lib_name] = (
          os.path.join(flags.cuda_lib_dir, cuda_library_file),
          os.path.join('/usr/lib/cuda/', cuda_library_file))
  return volume_specs


def main():
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--benchmark_configs_file', type=str, default=None, required=True,
      help='YAML file with benchmark configs.')
  parser.add_argument(
      '--benchmark_config_output', type=str, default=None, required=True,
      help='YAML file to store final config.')
  parser.add_argument(
      '--docker_image', type=str, default=None, required=True,
      help='Docker iage to use on K8S to run test.')
  parser.add_argument(
      '--cuda_lib_dir', type=str, default=None, required=False,
      help='Directory where cuda library files are located on gcloud node.')
  parser.add_argument(
      '--nvidia_lib_dir', type=str, default=None, required=False,
      help='Directory where nvidia library files are located on gcloud node.')

  flags, _ = parser.parse_known_args()
  logging.basicConfig(level=logging.DEBUG)

  config_base_path = os.path.dirname(__file__)

  config_text = open(
      os.path.join(config_base_path, flags.benchmark_configs_file), 'r').read()
  configs = yaml.load(config_text)

  # TODO(annarev): run benchmarks in parallel instead of sequentially.
  for config in configs:
    name = _ConvertToValidName(str(config['benchmark_name']))

    env_vars = {
        _TEST_NAME_ENV_VAR: name
    }
    gpu_count = (0 if 'gpus_per_machine' not in config
                 else config['gpus_per_machine'])
    volumes = {}
    if gpu_count > 0:
      volumes = _GetGpuVolumeMounts(flags)
      env_vars['LD_LIBRARY_PATH'] = (
          '/usr/lib/cuda:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu')

    env_vars.update(config.get('env_vars', {}))
    args = config.get('args', {})
    kubernetes_config = k8s_tensorflow_lib.GenerateConfig(
        config['worker_count'],
        config['ps_count'],
        _PORT,
        request_load_balancer=False,
        docker_image=flags.docker_image,
        name_prefix=name,
        additional_args=args,
        env_vars=env_vars,
        volumes=volumes,
        use_shared_volume=False,
        use_cluster_spec=False,
        gpu_limit=gpu_count)

    with open(flags.benchmark_config_output, 'w') as output_config_file:
      output_config_file.write(kubernetes_config)


if __name__ == '__main__':
  main()
