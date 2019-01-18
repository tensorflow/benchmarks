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
import importlib
import os
import time
import uuid

import perfzero.common.utils as utils
import perfzero.report.bench_result as bench_result
import perfzero.report.report as report


class BenchmarkRunner(object):
  """Executes tests and reports results."""

  def __init__(self):
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_dir = os.path.join(project_dir, 'workspace')
    site_packages_dir = os.path.join(workspace_dir, 'site-packages')
    auth_token_path = os.path.join(workspace_dir,
                                   'auth_tokens/benchmark_upload_gce.json')
    self.output_parent_dir = os.path.join(workspace_dir, 'output')
    """Setup environment before executing tests."""
    utils.check_and_print_env_var()
    utils.setup_python_path(site_packages_dir)
    utils.active_gcloud_service(auth_token_path)
    utils.make_dir_if_not_exist(self.output_parent_dir)

  def run_benchmark(self):
    """Run tests."""
    benchmark_methods = utils.get_env_var('ROGUE_TEST_METHODS').split(',')
    for benchmark_method in benchmark_methods:
      uid = str(uuid.uuid4())
      output_dir = os.path.join(self.output_parent_dir, uid)
      utils.make_dir_if_not_exist(output_dir)

      class_instance = self._instantiate_benchmark_class(output_dir)
      benchmark_id = '{}.{}'.format(class_instance.__class__.__name__,
                                    benchmark_method)

      start_time = time.time()
      # Run benchmark method
      getattr(class_instance, benchmark_method)()
      total_time = utils.get_milliseconds_diff(start_time)

      benchmark_result = class_instance.oss_report_object
      benchmark_result.total_time = total_time
      self._report_result(benchmark_id, benchmark_result, uid, output_dir)

  def _report_result(self, benchmark_id, benchmark_result, uid, output_dir):
    """Report results and upload artifacts."""
    project_name = utils.get_env_var('ROGUE_REPORT_PROJECT', default='LOCAL')

    # Upload benchmark ouput
    gcs_path = self._upload_benchmark_output(uid, output_dir)
    extras = {}
    extras['artifacts'] = gcs_path

    main_result, results, test_info, system_info = report.build_entry(
        benchmark_id,
        benchmark_result.total_time,
        test_environment=utils.get_env_var('ROGUE_TEST_ENV', default='local'),
        platform=utils.get_env_var('ROGUE_PLATFORM', default='unknown'),
        platform_type=utils.get_env_var(
            'ROGUE_PLATFORM_TYPE', default='unknown'))

    for test_result in benchmark_result.get_results():
      report.add_result(results, test_result)

    print('results:{}'.format(results))
    report.report_result(
        main_result,
        results,
        test_info,
        system_info,
        extras=extras,
        project=project_name,
        dev=False)

  def _upload_benchmark_output(self, uid, output_dir):
    """Uplaods artifacts to GCS."""
    target_code_dir = utils.get_env_var('ROGUE_CODE_DIR', default='not_set')

    if target_code_dir == 'not_set':
      print('Skipping uploading artifacts. ROGUE_CODE_DIR not set.')
      return 'not_set'

    gcs_path = '{}/{}/'.format(target_code_dir, uid)
    cmds = ['gsutil -m cp -r {}/* {}'.format(output_dir, gcs_path)]
    utils.run_commands(cmds)
    print('Uploaded benchmark output gcs at {}'.format(gcs_path))
    return gcs_path

  def _instantiate_benchmark_class(self, output_dir):
    """Returns initialized benchmark class."""
    module_import_path, class_name = utils.get_env_var(
        'ROGUE_TEST_CLASS').rsplit('.', 1)
    module = importlib.import_module(module_import_path)
    class_ = getattr(module, class_name)

    instance = class_(output_dir=output_dir)
    instance.oss_report_object = bench_result.BenchResult()
    return instance


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()

  benchmark_runner = BenchmarkRunner()
  benchmark_runner.run_benchmark()
