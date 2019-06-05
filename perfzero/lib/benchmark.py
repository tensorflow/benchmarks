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
"""Execute benchmark."""
from __future__ import print_function

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
import time

import perfzero.benchmark_method_runner as benchmark_method_runner
import perfzero.perfzero_config as perfzero_config
import perfzero.utils as utils


class BenchmarkRunner(object):
  """Execute benchmark and report results."""

  def __init__(self, config):
    self.config = config
    self.project_dir = os.path.abspath(
        os.path.dirname(os.path.dirname(__file__)))
    self.workspace_dir = os.path.join(self.project_dir, config.workspace)
    self.site_packages_dir = os.path.join(self.workspace_dir, 'site-packages')
    self.root_output_dir = os.path.join(self.workspace_dir, 'output')
    self.benchmark_execution_time = {}

  def _setup(self):
    """Download data and checkout git repository."""

    # Acticate gcloud service
    start_time = time.time()
    utils.setup_python_path(self.site_packages_dir, self.config.python_path_str)
    utils.active_gcloud_service(self.config.gcloud_key_file_url,
                                self.workspace_dir)
    utils.make_dir_if_not_exist(self.root_output_dir)
    self.benchmark_execution_time['activate_gcloud_service'] = (
        time.time() - start_time)

    # Download data
    start_time = time.time()
    utils.download_data(utils.parse_data_downloads_str(
        self.config.root_data_dir, self.config.gcs_downloads_str))
    utils.download_data(utils.parse_data_downloads_str(
        self.config.root_data_dir, self.config.data_downloads_str))
    self.benchmark_execution_time['download_data'] = time.time() - start_time

    # Checkout git repositories
    start_time = time.time()
    site_package_info = utils.checkout_git_repos(
        self.config.get_git_repos(self.site_packages_dir),
        self.config.use_cached_site_packages)
    self.benchmark_execution_time['checkout_repository'] = (
        time.time() - start_time)

    self.stream_handler = logging.StreamHandler(sys.stdout)
    self.stream_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(self.stream_handler)
    return site_package_info

  def _get_benchmark_methods(self):
    """Returns list of benchmark methods to execute."""
    filter_prefix = 'filter:'
    benchmark_methods = []
    for benchmark_method_pattern in self.config.benchmark_method_patterns:
      if filter_prefix not in benchmark_method_pattern:
        benchmark_methods.append(benchmark_method_pattern)
      else:
        index = benchmark_method_pattern.find(filter_prefix)
        benchmark_class = benchmark_method_pattern[:index - 1]
        pattern = benchmark_method_pattern[index + len(filter_prefix):]
        class_instance = utils.instantiate_benchmark_class(benchmark_class,
                                                           '/dev/null',
                                                           '')
        for benchmark_method_name in dir(class_instance):
          if re.match(pattern, benchmark_method_name):
            benchmark_methods.append(benchmark_class + '.' +
                                     benchmark_method_name)

    logging.info('The following benchmark methods will be executed: %s',
                 benchmark_methods)
    return benchmark_methods

  def run_benchmark(self):
    """Run benchmark."""
    harness_info = utils.get_git_repo_info(self.project_dir)
    site_package_info = self._setup()
    has_exception = False
    benchmark_success_results = {}
    benchmark_output_dirs = {}

    for benchmark_method in self._get_benchmark_methods():
      # Run the benchmark method in a separate process so that its memory usage
      # will not affect the execution of other benchmark method
      # This is a walkaround before we fix all memory leak issues in TensorFlow
      queue = multiprocessing.Queue()
      process = multiprocessing.Process(target=benchmark_method_runner.run,
                                        args=(benchmark_method,
                                              harness_info,
                                              site_package_info,
                                              self.root_output_dir,
                                              self.config, queue))
      process.start()
      process.join()
      method_has_exception, method_execution_time, succeeded, output_dir = queue.get()  # pylint: disable=line-too-long
      has_exception |= method_has_exception
      self.benchmark_execution_time[benchmark_method] = method_execution_time
      benchmark_success_results[benchmark_method] = succeeded
      benchmark_output_dirs[benchmark_method] = output_dir

    print('Benchmark execution time in seconds by operation:\n {}'.format(
        json.dumps(self.benchmark_execution_time, indent=2)))
    print('Benchmark success results:\n{}'.format(
        json.dumps(benchmark_success_results, indent=2)))
    print('Benchmark local output directories:\n{}'.format(
        json.dumps(benchmark_output_dirs, indent=2)))
    if has_exception:
      sys.exit(1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  perfzero_config.add_benchmark_parser_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()

  level = logging.DEBUG if FLAGS.debug else logging.INFO
  logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                      level=level)

  if unparsed:
    logging.error('Arguments %s are not recognized', unparsed)
    sys.exit(1)

  config_ = perfzero_config.PerfZeroConfig(mode='flags', flags=FLAGS)
  benchmark_runner = BenchmarkRunner(config_)
  benchmark_runner.run_benchmark()
