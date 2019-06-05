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
"""Upload test results."""
from __future__ import print_function

import json
import logging
import perfzero.utils as utils
import psutil
import socket

from six import u as unicode  # pylint: disable=W0622


def upload_execution_summary(bigquery_project_name, bigquery_dataset_table_name,
                             execution_summary):
  """Upload benchmark summary.

  Note: Using stream=False has a 1000 per day insert limit per table. Using
  stream=True, the documented limit is 50K+. With streaming there can be
  a small and possibly not noticeable delay to seeing the results the BigQuery
  UI, but there can be a 90 minute more or less delay in the results being part
  of exports.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    bigquery_project_name: Name of the gcp project.
    bigquery_dataset_table_name: data_set and table name.
    execution_summary: benchmark summary dictionary of results.
  """

  # pylint: disable=C6204
  import google.auth
  from google.cloud import bigquery

  if not bigquery_project_name:
    logging.info(
        'Skipped uploading benchmark result to bigquery because bigquery table name is not set.'
    )
    return

  if not bigquery_dataset_table_name:
    logging.info(
        'Skipped uploading benchmark result to bigquery because bigquery project name is not set.'
    )
    return

  credentials = google.auth.default()[0]
  dataset_name = bigquery_dataset_table_name.split('.')[0]
  table_name = bigquery_dataset_table_name.split('.')[1]
  client = bigquery.Client(
      project=bigquery_project_name, credentials=credentials)

  benchmark_summary_input = {}
  for key, value in execution_summary.items():
    if isinstance(value, dict):
      benchmark_summary_input[key] = unicode(json.dumps(value))
    else:
      benchmark_summary_input[key] = unicode(value)
  logging.debug('Bigquery input for benchmark_summary table is %s',
                json.dumps(benchmark_summary_input, indent=2))

  errors = []
  # TODO(tobyboyd): Shim to direct results to new table until all jobs
  # are updated.
  if 'benchmark_results' in dataset_name:
    if dataset_name == 'benchmark_results_dev':
      table_ref = client.dataset('perfzero_dev').table('benchmark_summary')
      table_obj = client.get_table(table_ref)
    elif dataset_name == 'benchmark_results':
      table_ref = client.dataset('perfzero').table('benchmark_summary')
      table_obj = client.get_table(table_ref)
  else:
    table_ref = client.dataset(dataset_name).table(table_name)
    table_obj = client.get_table(table_ref)

  errors.extend(client.insert_rows(table_obj, [benchmark_summary_input]))

  if errors:
    logging.error(
        'Failed to upload benchmark result to bigquery due to errors %s',
        errors)
  else:
    logging.info(
        'Uploaded benchmark result to the table %s of the bigquery project %s.',
        bigquery_dataset_table_name,
        bigquery_project_name)


def build_benchmark_result(raw_benchmark_result, has_exception):
  """Converts test_log.proto format to PerfZero format."""
  benchmark_result = {}
  benchmark_result['name'] = raw_benchmark_result['name']
  benchmark_result['wall_time'] = raw_benchmark_result['wall_time']

  succeeded = not has_exception
  extras = []
  for name in raw_benchmark_result.get('extras', {}):
    entry = {}
    entry['name'] = name

    if 'double_value' in raw_benchmark_result['extras'][name]:
      entry['value'] = raw_benchmark_result['extras'][name]['double_value']
    else:
      entry['value'] = raw_benchmark_result['extras'][name]['string_value']
    extras.append(entry)

  metrics = []
  for metric in raw_benchmark_result.get('metrics', []):
    value = metric['value']
    if 'min_value' in metric and metric['min_value'] > value:
      succeeded = False
    if 'max_value' in metric and metric['max_value'] < value:
      succeeded = False
    metrics.append(metric)

  benchmark_result['succeeded'] = succeeded
  benchmark_result['extras'] = extras
  benchmark_result['metrics'] = metrics

  return benchmark_result


def build_execution_summary(execution_timestamp, execution_id,
                            ml_framework_build_label, execution_label,
                            platform_name, system_name, output_gcs_url,
                            benchmark_result, env_vars, flags, harness_info,
                            site_package_info, process_info, has_exception):
  """Builds summary of the execution."""
  # Avoids module not found during setup phase when tf is not installed yet.
  # pylint: disable=C6204
  import tensorflow as tf

  benchmark_info = {}
  benchmark_info['harness_name'] = 'perfzero'
  benchmark_info['harness_info'] = harness_info
  benchmark_info['has_exception'] = has_exception
  if execution_label:
    benchmark_info['execution_label'] = execution_label
  if output_gcs_url:
    benchmark_info['output_url'] = '{}/{}/'.format(output_gcs_url, execution_id)
  if env_vars:
    benchmark_info['env_vars'] = env_vars
  if flags:
    benchmark_info['flags'] = flags
  benchmark_info['site_package_info'] = site_package_info

  ml_framework_info = {}
  ml_framework_info['name'] = 'tensorflow'
  ml_framework_info['version'] = tf.__version__
  # tf.__git_version__ in Python3 has format b'version_string'
  if tf.__git_version__[0] == 'b':
    ml_framework_info['build_version'] = tf.__git_version__[2:-1]
  else:
    ml_framework_info['build_version'] = tf.__git_version__

  if ml_framework_build_label:
    ml_framework_info['build_label'] = ml_framework_build_label

  system_info = {}
  gpu_info = utils.get_gpu_info()
  if platform_name:
    system_info['platform_name'] = platform_name
  if system_name:
    system_info['system_name'] = system_name
  system_info['accelerator_driver_version'] = gpu_info['gpu_driver_version']
  system_info['accelerator_model'] = gpu_info['gpu_model']
  system_info['accelerator_count'] = gpu_info['gpu_count']
  system_info['cpu_model'] = utils.get_cpu_name()
  system_info['physical_cpu_count'] = psutil.cpu_count(logical=False)
  system_info['logical_cpu_count'] = psutil.cpu_count(logical=True)
  system_info['cpu_socket_count'] = utils.get_cpu_socket_count()
  system_info['hostname'] = socket.gethostname()

  execution_summary = {}
  execution_summary['execution_id'] = execution_id
  execution_summary['execution_timestamp'] = execution_timestamp
  execution_summary['benchmark_result'] = benchmark_result
  execution_summary['benchmark_info'] = benchmark_info
  execution_summary['setup_info'] = {}
  execution_summary['ml_framework_info'] = ml_framework_info
  execution_summary['system_info'] = system_info
  if process_info:
    execution_summary['process_info'] = process_info

  return execution_summary
