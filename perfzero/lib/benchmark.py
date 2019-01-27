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
import time
import uuid

import perfzero.common.utils as utils
import perfzero.common.perfzero_config as perfzero_config
import perfzero.report.report_utils as report_utils
import perfzero.report.benchmark_result as benchmark_result


class BenchmarkRunner(object):
  """Executes tests and reports results."""

  def __init__(self, config=None):
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_dir = os.path.join(project_dir, 'workspace')
    site_packages_dir = os.path.join(workspace_dir, 'site-packages')
    auth_token_path = os.path.join(workspace_dir,
                                   'auth_tokens/benchmark_upload_gce.json')
    self.output_root_dir = os.path.join(workspace_dir, 'output')
    self.config = config
    """Setup environment before executing tests."""
    utils.setup_python_path(site_packages_dir, config.python_paths_str)
    utils.active_gcloud_service(auth_token_path)
    utils.make_dir_if_not_exist(self.output_root_dir)

  def run_benchmark(self):
    """Run tests."""
    for benchmark_method in self.config.benchmark_methods_str.split(','):
      uid = str(uuid.uuid4())
      output_dir = os.path.join(self.output_root_dir, uid)
      utils.make_dir_if_not_exist(output_dir)

      class_instance = self._instantiate_benchmark_class(output_dir)
      class_method_name = '{}.{}'.format(class_instance.__class__.__name__,
                                         benchmark_method)

      start_time = time.time()
      # Run benchmark method
      getattr(class_instance, benchmark_method)()
      total_time = utils.get_milliseconds_diff(start_time)

      oss_report_object = class_instance.oss_report_object
      oss_report_object.total_time = total_time
      self._report_result(class_method_name, oss_report_object, uid, output_dir)

  def _report_result(self, class_method_name, oss_report_object, uid,
                     output_dir):
    """Report results and upload artifacts."""
    # Upload benchmark ouput
    output_gcs_dir = None
    if not self.config.output_gcs_url_str:
      print('Skipping uploading output. output_gcs_url_str is not set.')
    elif not os.listdir(output_dir):
      print('Skipping uploading output. Directory is empty:{}'.
            format(output_dir))
    else:
      output_gcs_dir = '{}/{}/'.format(self.config.output_gcs_url_str, uid)
      utils.upload_to_gcs(output_dir, output_gcs_dir)

    extras = {}
    extras['artifacts'] = output_gcs_dir

    main_result, results, test_info, system_info = report_utils.build_entry(
        class_method_name,
        total_time=oss_report_object.total_time,
        test_environment=self.config.test_env_str,
        platform=self.config.platform_name_str,
        platform_type=self.config.platform_type_str)

    for test_result in oss_report_object.get_results():
      report_utils.add_result(results, test_result)

    print('results:{}'.format(results))
    report_utils.report_result(
        main_result,
        results,
        test_info,
        system_info,
        extras=extras,
        project=self.config.project_name_str,
        dev=False)

  def _instantiate_benchmark_class(self, output_dir):
    """Return initialized benchmark class."""
    module_import_path, class_name = self.config.test_class_str.rsplit('.', 1)
    module = importlib.import_module(module_import_path)
    class_ = getattr(module, class_name)

    instance = class_(output_dir=output_dir)
    instance.oss_report_object = benchmark_result.BenchmarkResult()
    return instance


if __name__ == '__main__':

  config = perfzero_config.PerfZeroConfig(mode='env')
  benchmark_runner = BenchmarkRunner(config)
  benchmark_runner.run_benchmark()
