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

import importlib
import os
import json
import sys
import time
import uuid
import re
import argparse
import logging
import datetime
import traceback

import perfzero.utils as utils
import perfzero.perfzero_config as perfzero_config
import perfzero.report_utils as report_utils


class BenchmarkRunner(object):
  """Execute benchmark and report results."""

  def __init__(self, config=None):
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    self.workspace_dir = os.path.join(project_dir, 'workspace')
    self.site_packages_dir = os.path.join(self.workspace_dir, 'site-packages')
    self.auth_token_path = os.path.join(
        self.workspace_dir, 'auth_tokens/benchmark_upload_gce.json')
    self.output_root_dir = os.path.join(self.workspace_dir, 'output')
    self.config = config
    self._setup()

  def _setup(self):
    utils.setup_python_path(self.site_packages_dir, config.python_path_str)
    utils.active_gcloud_service(self.auth_token_path)
    utils.make_dir_if_not_exist(self.output_root_dir)

    self.streamHandler = logging.StreamHandler(sys.stdout)
    self.streamHandler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(self.streamHandler)

  def _get_benchmark_methods(self):
    filter_prefix = 'filter:'
    benchmark_methods = []
    for benchmark_method_maybe_filter in self.config.benchmark_methods_maybe_filter:
      if filter_prefix not in benchmark_method_maybe_filter:
        benchmark_methods.append(benchmark_method_maybe_filter)
      else:
        index = benchmark_method_maybe_filter.find(filter_prefix)
        benchmark_class = benchmark_method_maybe_filter[:index - 1]
        pattern = benchmark_method_maybe_filter[index + len(filter_prefix):]
        class_instance = self._instantiate_benchmark_class(benchmark_class, '/dev/null')
        for benchmark_method_name in dir(class_instance):
          if re.match(pattern, benchmark_method_name):
            benchmark_methods.append(benchmark_class + '.' + benchmark_method_name)

    return benchmark_methods

  def run_benchmark(self):
    """Run benchmark."""
    for benchmark_method in self._get_benchmark_methods():
      try:
        execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        execution_timestamp = time.time()
        output_dir = os.path.join(self.output_root_dir, execution_id)
        utils.make_dir_if_not_exist(output_dir)
        has_exception = False
        benchmark_class, benchmark_method_name = benchmark_method.rsplit('.', 1)
        benchmark_module, benchmark_class_name = benchmark_class.rsplit('.', 1)

        # Setup per-method file logger
        filehandler = logging.FileHandler(
            filename=os.path.join(output_dir, 'perfzero.log'), mode='w')
        filehandler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logging.getLogger().addHandler(filehandler)
        class_instance = self._instantiate_benchmark_class(benchmark_class, output_dir)

        # tf.test.Benchmark.report_benchmark() will write benchmark results to
        # the file whose path is benchmark_result_file_path_prefix + benchmark_method
        benchmark_result_file_path_prefix = os.path.join(output_dir, 'proto_')
        os.environ['TEST_REPORT_FILE_PREFIX'] = benchmark_result_file_path_prefix
        benchmark_result_file_path = benchmark_result_file_path_prefix + \
                                     benchmark_class_name + "." + benchmark_method_name

        # Run benchmark method
        logging.info('Start benchmark: %s', benchmark_method)
        getattr(class_instance, benchmark_method_name)()
        logging.info('End benchmark: %s', benchmark_method)
        # Read and build benchmark results
        raw_benchmark_result = utils.read_benchmark_result(benchmark_result_file_path)
      except Exception as e:
        logging.error('Benchmark execution for %s failed due to error:\n %s',
                      benchmark_method, traceback.format_exc())
        has_exception = True
        raw_benchmark_result = {}
        raw_benchmark_result['name'] = benchmark_method
        raw_benchmark_result['wall_time'] = -1
        raw_benchmark_result['extras'] = {}
      finally:
        with open(os.path.join(self.workspace_dir, 'setup_info.log'), 'r') as f:
          setup_info = json.load(f)
        benchmark_result = report_utils.build_benchmark_result(raw_benchmark_result)
        execution_summary = report_utils.build_execution_summary(
            execution_timestamp, execution_id, self.config.test_env_str,
            self.config.ml_framework_build_label_str,
            self.config.execution_label_str, self.config.platform_name_str,
            self.config.system_name_str, self.config.output_gcs_url_str,
            benchmark_result, self.config.get_env_vars(), setup_info, has_exception)
        report_utils.upload_execution_summary(
            self.config.bigquery_project_name_str,
            self.config.bigquery_dataset_table_name_str, execution_summary,
            raw_benchmark_result)
        logging.info('Benchmark execution completed with summary:\n %s',
                     json.dumps(execution_summary, indent=2))
        utils.maybe_upload_to_gcs(output_dir, self.config.output_gcs_url_str)
        logging.getLogger().removeHandler(filehandler)

  def _instantiate_benchmark_class(self, benchmark_class, output_dir):
    """Return initialized benchmark class."""
    module_import_path, class_name = benchmark_class.rsplit('.', 1)
    module = importlib.import_module(module_import_path)
    class_ = getattr(module, class_name)
    instance = class_(output_dir=output_dir)

    return instance


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', action='store_true')
  FLAGS, unparsed = parser.parse_known_args()

  level = logging.INFO
  if FLAGS.debug:
    level = logging.DEBUG
  logging.basicConfig(
      format='%(asctime)s %(levelname)s: %(message)s', level=level)

  config = perfzero_config.PerfZeroConfig(mode='env')
  benchmark_runner = BenchmarkRunner(config)
  benchmark_runner.run_benchmark()
