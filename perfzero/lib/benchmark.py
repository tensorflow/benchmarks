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

import perfzero.utils as utils
import perfzero.perfzero_config as perfzero_config
import perfzero.report_utils as report_utils


class BenchmarkRunner(object):
  """Execute benchmark and report results."""

  def __init__(self, config=None):
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_dir = os.path.join(project_dir, 'workspace')
    self.site_packages_dir = os.path.join(workspace_dir, 'site-packages')
    self.auth_token_path = os.path.join(
        workspace_dir, 'auth_tokens/benchmark_upload_gce.json')
    self.output_root_dir = os.path.join(workspace_dir, 'output')
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
    # Check if filter or list of exact methods to execute.
    methods_str = self.config.benchmark_methods_str

    if not methods_str.startswith(filter_prefix):
      return methods_str.split(',')

    # Instantiate benchmark class to find methods that match the given pattern
    methods_matching_pattern = []
    pattern = methods_str[len(filter_prefix):]
    class_instance = self._instantiate_benchmark_class('/dev/null')
    for method in dir(class_instance):
      if re.match(pattern, method):
        methods_matching_pattern.append(method)

    logging.info('Methods {} matched the pattern {}'.format(
        methods_matching_pattern, pattern))
    return methods_matching_pattern

  def run_benchmark(self):
    """Run benchmark."""
    for benchmark_method in self._get_benchmark_methods():
      try:
        execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        output_dir = os.path.join(self.output_root_dir, execution_id)
        utils.make_dir_if_not_exist(output_dir)

        # Setup per-method file logger
        filehandler = logging.FileHandler(
            filename=os.path.join(output_dir, 'perfzero.log'), mode='w')
        filehandler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logging.getLogger().addHandler(filehandler)

        class_instance = self._instantiate_benchmark_class(output_dir)
        benchmark_name = '{}.{}'.format(class_instance.__class__.__name__,
                                        benchmark_method)

        # tf.test.Benchmark.report_benchmark() will write benchmark results to
        # the file whose path is benchmark_result_file_path_prefix +
        # benchmark_name
        benchmark_result_file_path_prefix = os.path.join(output_dir, 'proto_')
        os.environ[
            'TEST_REPORT_FILE_PREFIX'] = benchmark_result_file_path_prefix
        benchmark_result_file_path = benchmark_result_file_path_prefix + benchmark_name

        # Run benchmark method
        logging.info('Start benchmark: %s', benchmark_name)
        getattr(class_instance, benchmark_method)()
        logging.info('End benchmark: %s', benchmark_name)
        # Read and upload benchmark results
        benchmark_result = utils.read_benchmark_result(
            benchmark_result_file_path)
        self._upload_execution_summary(benchmark_result, execution_id,
                                       output_dir)

      finally:
        logging.getLogger().removeHandler(filehandler)

  def _upload_execution_summary(self, benchmark_result, execution_id,
                                output_dir):
    """Report results and upload artifacts."""
    # Upload benchmark ouput
    output_gcs_dir_with_uid = ''
    if not self.config.output_gcs_url_str:
      logging.info(
          'Skipped uploading output because output_gcs_url_str is not set.')
    elif not os.listdir(output_dir):
      logging.info(
          'Skipped uploading output because there is no file in directory %s',
          output_dir)
    else:
      output_gcs_dir_with_uid = '{}/{}/'.format(self.config.output_gcs_url_str,
                                                execution_id)
      utils.upload_to_gcs(output_dir, self.config.output_gcs_url_str)

    execution_summary = report_utils.build_execution_summary(
        execution_id, self.config.test_env_str, self.config.platform_name_str,
        self.config.system_name_str, output_gcs_dir_with_uid, benchmark_result)

    logging.info('Benchmark summary is %s',
                 json.dumps(execution_summary, indent=2))

    report_utils.upload_execution_summary(self.config.bigquery_project_name_str,
                                          self.config.bigquery_table_name_str,
                                          execution_summary)

  def _instantiate_benchmark_class(self, output_dir):
    """Return initialized benchmark class."""
    module_import_path, class_name = self.config.benchmark_class_str.rsplit(
        '.', 1)
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
