"""Run benchmarks from kubernetes_config_file yml file.

This script should only be run from opensource repository.
"""
import argparse
import logging
import sys

import kubectl_util


def _RunBenchmark(name, yaml_file):
  """Runs a single distributed benchmark.

  Args:
    name: name of the benchmark to run.
    yaml_file: path to kubernetes config file.

  Returns:
    Success signal.
  """
  kubectl_util.DeletePods(name, yaml_file)
  kubectl_util.CreatePods(name, yaml_file)
  success = kubectl_util.WaitForCompletion(name)
  kubectl_util.DeletePods(name, yaml_file)
  return success


def main():
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument(
      '--kubernetes_config_file', type=str, default=None, required=True,
      help=('Kubernetes YAML config file specifying TensorFlow worker and '
            'parameter server jobs to bring up.'))
  parser.add_argument(
      '--benchmark_name', type=str, default=None, required=True,
      help=('Benchmark name. Can only contain alphanumeric characters and '
            'dashes'))
  flags, _ = parser.parse_known_args()
  logging.basicConfig(level=logging.DEBUG)

  success = _RunBenchmark(flags.benchmark_name, flags.kubernetes_config_file)
  if not success:
    sys.exit(1)


if __name__ == '__main__':
  main()
