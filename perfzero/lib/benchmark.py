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
import datetime
import importlib
import json
import logging
import os
import re
import sys
import time
import traceback

import perfzero.device_utils as device_utils
import perfzero.perfzero_config as perfzero_config
import perfzero.report_utils as report_utils
import perfzero.utils as utils


class BenchmarkRunner(object):
  """Execute benchmark and report results."""

  def __init__(self, config):
    self.config = config
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    self.workspace_dir = os.path.join(project_dir, config.workspace)
    self.site_packages_dir = os.path.join(self.workspace_dir, 'site-packages')
    self.auth_token_path = os.path.join(
        self.workspace_dir, 'auth_tokens/benchmark_upload_gce.json')
    self.root_output_dir = os.path.join(self.workspace_dir, 'output')
    self.benchmark_execution_time = {}

  def _setup(self):
    """Download data and checkout git repository."""
    # Set up the raid array.
    start_time = time.time()
    device_utils.create_drive_from_devices('/data',
                                           self.config.gce_nvme_raid_str)
    self.benchmark_execution_time['create_drive'] = time.time() - start_time

    start_time = time.time()
    utils.download_from_gcs([{'gcs_url': 'gs://tf-performance/auth_tokens',
                              'local_path': os.path.join(self.workspace_dir, 'auth_tokens')}])  # pylint: disable=line-too-long
    self.benchmark_execution_time['download_token'] = time.time() - start_time

    # Acticate gcloud service
    start_time = time.time()
    utils.setup_python_path(self.site_packages_dir, self.config.python_path_str)
    utils.active_gcloud_service(self.auth_token_path)
    utils.make_dir_if_not_exist(self.root_output_dir)
    self.benchmark_execution_time['activate_gcloud_service'] = time.time() - start_time  # pylint: disable=line-too-long

    # Download data
    start_time = time.time()
    utils.download_from_gcs(self.config.get_gcs_downloads('/data'))
    self.benchmark_execution_time['download_data'] = time.time() - start_time

    # Checkout git repositories
    start_time = time.time()
    site_package_info = utils.checkout_git_repos(
        self.config.get_git_repos(self.site_packages_dir),
        self.config.force_update)
    self.benchmark_execution_time['checkout_repository'] = time.time() - start_time  # pylint: disable=line-too-long

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
        class_instance = self._instantiate_benchmark_class(benchmark_class,
                                                           '/dev/null')
        for benchmark_method_name in dir(class_instance):
          if re.match(pattern, benchmark_method_name):
            benchmark_methods.append(benchmark_class + '.' +
                                     benchmark_method_name)
    return benchmark_methods

  def run_benchmark(self):
    """Run benchmark."""
    site_package_info = self._setup()
    has_exception = False
    benchmark_success_results = {}

    for benchmark_method in self._get_benchmark_methods():
      start_timestamp = time.time()
      method_has_exception = False
      execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
      execution_timestamp = time.time()
      output_dir = os.path.join(self.root_output_dir, execution_id)
      utils.make_dir_if_not_exist(output_dir)
      benchmark_class, benchmark_method_name = benchmark_method.rsplit('.', 1)
      benchmark_class_name = benchmark_class.rsplit('.', 1)[1]

      # Setup per-method file logger
      filehandler = logging.FileHandler(
          filename=os.path.join(output_dir, 'perfzero.log'), mode='w')
      filehandler.setFormatter(
          logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
      logging.getLogger().addHandler(filehandler)

      try:
        class_instance = self._instantiate_benchmark_class(benchmark_class,
                                                           output_dir)
        # tf.test.Benchmark.report_benchmark() writes results to a file with
        # path benchmark_result_file_path_prefix + benchmark_method
        benchmark_result_file_path_prefix = os.path.join(output_dir, 'proto_')
        os.environ['TEST_REPORT_FILE_PREFIX'] = benchmark_result_file_path_prefix  # pylint: disable=line-too-long
        benchmark_result_file_path = '{}{}.{}'.format(
            benchmark_result_file_path_prefix,
            benchmark_class_name,
            benchmark_method_name)
        # Run benchmark method
        logging.info('Start benchmark: %s', benchmark_method)
        getattr(class_instance, benchmark_method_name)()
        logging.info('End benchmark: %s', benchmark_method)
        # Read and build benchmark results
        raw_benchmark_result = utils.read_benchmark_result(benchmark_result_file_path)  # pylint: disable=line-too-long
        # Explicitly overwrite the name to be the full path to benchmark method
        raw_benchmark_result['name'] = benchmark_method
      except Exception:  # pylint: disable=W0703
        logging.error('Benchmark execution for %s failed due to error:\n %s',
                      benchmark_method, traceback.format_exc())
        method_has_exception = True
        has_exception = True
        raw_benchmark_result = {}
        raw_benchmark_result['name'] = benchmark_method
        raw_benchmark_result['wall_time'] = -1
        raw_benchmark_result['extras'] = {}

      upload_timestamp = time.time()
      benchmark_result = report_utils.build_benchmark_result(
          raw_benchmark_result, method_has_exception)
      benchmark_success_results[benchmark_method] = benchmark_result['succeeded']  # pylint: disable=line-too-long
      execution_summary = report_utils.build_execution_summary(
          execution_timestamp,
          execution_id,
          self.config.ml_framework_build_label_str,
          self.config.execution_label_str,
          self.config.platform_name_str,
          self.config.system_name_str,
          self.config.output_gcs_url_str,
          benchmark_result,
          self.config.get_env_vars(),
          self.config.get_flags(),
          site_package_info,
          method_has_exception)
      report_utils.upload_execution_summary(
          self.config.bigquery_project_name_str,
          self.config.bigquery_dataset_table_name_str,
          execution_summary)
      logging.info('Benchmark execution for %s completed with summary:\n %s',
                   benchmark_method, json.dumps(execution_summary, indent=2))
      utils.maybe_upload_to_gcs(output_dir, self.config.output_gcs_url_str)
      logging.getLogger().removeHandler(filehandler)
      self.benchmark_execution_time[benchmark_method] = {}
      self.benchmark_execution_time[benchmark_method]['benchmark_time'] = upload_timestamp - start_timestamp  # pylint: disable=line-too-long
      self.benchmark_execution_time[benchmark_method]['upload_time'] = time.time() - upload_timestamp  # pylint: disable=line-too-long

    print('Benchmark execution time in seconds by operation:\n {}'.format(
        json.dumps(self.benchmark_execution_time, indent=2)))
    print('benchmark success results:\n{}'.format(
        json.dumps(benchmark_success_results, indent=2)))
    if has_exception:
      sys.exit(1)

  def _instantiate_benchmark_class(self, benchmark_class, output_dir):
    """Return initialized benchmark class."""
    module_import_path, class_name = benchmark_class.rsplit('.', 1)
    module = importlib.import_module(module_import_path)
    class_ = getattr(module, class_name)
    instance = class_(output_dir=output_dir)

    return instance


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  perfzero_config.add_parser_arguments(parser)
  parser.add_argument(
      '--debug',
      action='store_true',
      help='If set, use debug level logging. Otherwise, use info level logging')
  FLAGS, unparsed = parser.parse_known_args()

  level = logging.INFO
  if FLAGS.debug:
    level = logging.DEBUG
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=level)

  if unparsed:
    logging.warn('Arguments %s are not recognized', unparsed)

  config_ = perfzero_config.PerfZeroConfig(mode='flags', flags=FLAGS)
  benchmark_runner = BenchmarkRunner(config_)
  benchmark_runner.run_benchmark()
