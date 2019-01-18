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

"""Reports results of tests."""
from __future__ import print_function

import perfzero.common.cpu as cpu_info
import perfzero.common.upload.result_info as result_info
import perfzero.common.upload.result_upload as result_upload
import perfzero.common.utils as utils


def report_result(main_result,
                  results,
                  test_info,
                  system_info,
                  project=None,
                  extras=None,
                  dev=True):
  """Reports test results.

  Args:
    main_result: Unique id of the test with in mlperf tests.
    results: total test time.
    test_info: kokoro, roguezero or whatever
    system_info: gcp, dgx, aws
    project: Cloud project. Set to 'LOCAL' to only print to console.
    extras: dictionary of extra info to store
    dev: True to use development environment.
  """
  dataset = 'benchmark_results_dev'
  if not dev:
    dataset = 'benchmark_results'

  result_upload.upload_result(
      main_result,
      results,
      project,
      test_info=test_info,
      system_info=system_info,
      extras=extras,
      dataset=dataset)
  print('RogueZero Reported to:{}:{}'.format(project, dataset))


def build_entry(test_id,
                total_time,
                group_run_id=None,
                test_environment=None,
                platform=None,
                platform_type=None,
                channel='NIGHTLY',
                build_type='OTB',
                model='unknown',
                test_cmd='unknown',
                accel_type=None,
                accel_cnt=None):
  """Builds result row.

  Args:
    test_id: Unique id of the test with in mlperf tests.
    total_time: total test time.
    group_run_id: Provide if run should be aggregated.
    test_environment: kokoro, roguezero or whatever
    platform: gcp, dgx, aws
    platform_type: n1.standard-64, base image used.
    channel: NIGHTLY, RC, RELEASE, CUSTOM, or whatever.
    build_type: OTB (out of the box), CUDA_10, or whatever.
    model: model being tested, used to group results.
    test_cmd: Command run to start the test.
    accel_type: Type of accelerator, if None will try to guess.
    accel_cnt: Number of accelerators, if None will try to guess.

  Returns:
    main_result, results, test_info, and system_info.
  """
  framework = 'tensorflow'

  # Pulls CPU Info.
  cpu_type, cpu_sockets, cpu_cores, _ = cpu_info.get_cpu_info()
  # cpu_type, cpu_sockets, cpu_cores, _ = 'unknown', 1, 1, None

  # Pulls NVIDIA GPU Info.
  if not accel_type:
    _, accel_type = utils.get_gpu_info()
    if not accel_cnt:
      accel_cnt = utils.get_gpu_count()

  # Pulls tensorflow version info.version.
  tf_ver, tf_git_version = utils.get_tf_full_version()
  # Python3 tf_git_version is wrapped with b'blahblah'
  if tf_git_version.index('b') == 0:
    print('Clean __git_version__ {}'.format(tf_git_version))
    tf_git_version = tf_git_version[2:-1]

  system_info = result_info.build_system_info(
      platform=platform,
      platform_type=platform_type,
      accel_type=accel_type,
      cpu_type=cpu_type,
      cpu_cores=cpu_cores,
      cpu_sockets=cpu_sockets)

  # Main result config
  main_result, results = result_info.build_test_result(
      test_id,
      total_time,
      result_units='ms',
      result_type='total_time',
      test_harness='perfzero',
      test_environment=test_environment)

  test_info = result_info.build_test_info(
      framework=framework,
      framework_version=tf_ver,
      framework_describe=tf_git_version,
      channel=channel,
      build_type=build_type,
      model=model,
      cmd=test_cmd,
      accel_cnt=accel_cnt,
      group_run_id=group_run_id)

  return main_result, results, test_info, system_info


def add_result(results, result):
  """Add result to results list."""
  result_info.build_result_info(
      results,
      result['result'],
      result_type=result['result_type'],
      result_units=result['result_unit'],
      expected_min=result.get('expected_min'),
      expected_max=result.get('expected_max'),
      result_status=result.get('result_status'))
