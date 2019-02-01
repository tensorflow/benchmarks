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

import copy
import datetime
import json
import os
import pwd
import uuid
import pytz
import google.auth
import logging
from google.cloud import bigquery
from google.cloud.bigquery.dbapi import connect
from six import u as unicode  # pylint: disable=W0622

import perfzero.utils as utils


def upload_execution_summary(bigquery_project_name, bigquery_table_name,
                             execution_summary):
  """Upload benchmark summary.

  Note: Using stream=False has a 1000 per day insert limit per table. Using
  stream=True, the documented limit is 50K+. With streaming there can be
  a small and possibly not noticeable delay to seeing the results the BigQuery
  UI, but there can be a 90 minute more or less delay in the results being part
  of exports.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.
  """

  credentials = google.auth.default()[0]

  sql = """INSERT into {} (result_id, test_id, test_harness,
           test_environment, result_info, user, timestamp, system_info,
           test_info, extras)
             VALUES
           (@result_id, @test_id, @test_harness, @test_environment,
           @result_info, @user, @timestamp, @system_info, @test_info, @extras)
           """.format(bigquery_table_name)

  entry = _translate_summary_into_old_bigquery_format(execution_summary,
                                                      credentials)
  logging.info('Bigquery entry is {}'.format(json.dumps(entry, indent=2)))

  if not bigquery_project_name:
    logging.info(
        'Skipped uploading benchmark result to bigquery because bigquery project id is not set.'
    )
    return

  client = bigquery.Client(
      project=bigquery_project_name, credentials=credentials)
  conn = connect(client=client)
  cursor = conn.cursor()
  cursor.execute(sql, parameters=entry)
  conn.commit()
  cursor.close()
  conn.close()
  logging.info(
      'Uploaded benchmark result to the table {} of the bigquery project {}.'
      .format(bigquery_table_name, bigquery_project_name))


def _translate_summary_into_old_bigquery_format(execution_summary, credentials):
  entry = {}
  entry['result_id'] = unicode(execution_summary['execution_id'])
  entry['test_id'] = unicode(execution_summary['benchmark_result']['name'])
  entry['test_harness'] = unicode('perfzero')
  entry['test_environment'] = unicode(
      execution_summary['benchmark_info']['test_environment'])

  result_info_list = []
  result_info = {}
  result_info[
      'result'] = execution_summary['benchmark_result']['wall_time'] * 1000
  result_info['result_type'] = 'total_time'
  result_info['result_unit'] = 'ms'
  result_info_list.append(result_info)
  # Extra fields whose value is a json-formatted string
  if 'accuracy' in execution_summary['benchmark_result']['extras']:
    attributes = json.loads(execution_summary['benchmark_result']['extras']
                            ['accuracy']['string_value'])
    result_info = {}
    result_info['result'] = attributes['value']
    result_info['result_type'] = 'quality'
    result_info['result_unit'] = 'top_1'
    if 'succeeded' in attributes:
      if attributes['succeeded'] == True:
        result_info['result_status'] = 'PASS'
      else:
        result_info['result_status'] = 'FAILED'
    result_info_list.append(result_info)
  if 'accuracy_top_5' in execution_summary['benchmark_result']['extras']:
    attributes = json.loads(execution_summary['benchmark_result']['extras']
                            ['accuracy']['string_value'])
    result_info = {}
    result_info['result'] = attributes['value']
    result_info['result_type'] = 'quality'
    result_info['result_unit'] = 'top_5'
    result_info_list.append(result_info)
  if 'top_1_train_accuracy' in execution_summary['benchmark_result']['extras']:
    attributes = json.loads(execution_summary['benchmark_result']['extras']
                            ['top_1_train_accuracy']['string_value'])
    result_info = {}
    result_info['result'] = attributes['value']
    result_info['result_type'] = 'quality'
    result_info['result_unit'] = 'top_1_train_accuracy'
    result_info_list.append(result_info)
  if 'exp_per_second' in execution_summary['benchmark_result']['extras']:
    attributes = json.loads(execution_summary['benchmark_result']['extras']
                            ['exp_per_second']['string_value'])
    result_info = {}
    result_info['result'] = attributes['value']
    result_info['result_type'] = 'exp_per_second'
    result_info['result_unit'] = 'exp_per_second'
    result_info_list.append(result_info)
  if 'avg_exp_per_second' in execution_summary['benchmark_result']['extras']:
    attributes = json.loads(execution_summary['benchmark_result']['extras']
                            ['avg_exp_per_second']['string_value'])
    result_info = {}
    result_info['result'] = attributes['value']
    result_info['result_type'] = 'avg_exp_per_second'
    result_info['result_unit'] = 'exp_per_second'
    result_info_list.append(result_info)

  # Extra fields whose value is double
  if 'top_1_accuracy' in execution_summary['benchmark_result']['extras']:
    result_info = {}
    result_info['result'] = execution_summary['benchmark_result']['extras'][
        'top_1_accuracy']['double_value']
    result_info['result_type'] = 'quality'
    result_info['result_unit'] = 'top_1'
    result_info_list.append(result_info)
  if 'top_5_accuracy' in execution_summary['benchmark_result']['extras']:
    result_info = {}
    result_info['result'] = execution_summary['benchmark_result']['extras'][
        'top_5_accuracy']['double_value']
    result_info['result_type'] = 'quality'
    result_info['result_unit'] = 'top_5'
    result_info_list.append(result_info)

  entry['result_info'] = unicode(json.dumps(result_info_list))

  if hasattr(credentials, 'service_account_email'):
    entry['user'] = unicode(credentials.service_account_email)
  else:
    entry['user'] = unicode(pwd.getpwuid(os.getuid())[0])

  # gpylint warning suggests using a different lib that does not look helpful.
  # pylint: disable=W6421
  entry['timestamp'] = unicode(
      str(datetime.datetime.utcnow().replace(tzinfo=pytz.utc)))

  system_info = {}
  system_info['platform'] = execution_summary['system_info']['platform_name']
  system_info['platform_type'] = execution_summary['system_info']['system_name']
  system_info['accel_type'] = execution_summary['system_info']['gpu_model']
  system_info['accel_driver_version'] = execution_summary['system_info'][
      'gpu_driver_version']
  system_info['cpu_cores'] = execution_summary['system_info']['cpu_core_count']
  system_info['cpu_type'] = execution_summary['system_info']['cpu_model']
  system_info['cpu_sockets'] = execution_summary['system_info'][
      'cpu_socket_count']
  entry['system_info'] = unicode(json.dumps(system_info))

  test_info = {}
  test_info['framework'] = execution_summary['ml_framework_info']['framework']
  test_info['channel'] = execution_summary['ml_framework_info']['channel']
  test_info['build_type'] = execution_summary['ml_framework_info']['build_type']
  test_info['accel_cnt'] = execution_summary['system_info']['gpu_count']
  test_info['framework_version'] = execution_summary['ml_framework_info'][
      'version']
  test_info['framework_describe'] = execution_summary['ml_framework_info'][
      'git_version']
  test_info['model'] = 'unknown'
  test_info['cmd'] = 'unknown'
  entry['test_info'] = unicode(json.dumps(test_info))

  extras = {}
  extras['artifact'] = execution_summary['benchmark_info']['output_url']
  entry['extras'] = unicode(json.dumps(extras))

  return entry


#def upload_with_stream_mode(client, dataset, table, row):
#  """Uploads row to BigQuery via streaming interface."""
#  table_ref = client.dataset(dataset).table(table)
#  table_obj = client.get_table(table_ref)  # API request
#  errors = client.insert_rows(table_obj, [row])


def build_execution_summary(execution_id, test_environment, platform_name,
                            system_name, output_url, benchmark_result):
  import tensorflow as tf

  benchmark_info = {}
  benchmark_info['test_environment'] = test_environment
  benchmark_info['date_time'] = unicode(
      datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
  benchmark_info['output_url'] = output_url

  ml_framework_info = {}
  ml_framework_info['framework'] = 'tensorflow'
  ml_framework_info['version'] = tf.__version__
  # tf.__git_version__ in Python3 has format b'version_string'
  if tf.__git_version__[0] == 'b':
    ml_framework_info['git_version'] = tf.__git_version__[2:-1]
  else:
    ml_framework_info['git_version'] = tf.__git_version__
  ml_framework_info['channel'] = 'NIGHTLY'
  ml_framework_info['build_type'] = 'OTB'

  system_info = {}
  gpu_info = utils.get_gpu_info()
  system_info['platform_name'] = platform_name
  system_info['system_name'] = system_name
  system_info['gpu_driver_version'] = gpu_info['gpu_driver_version']
  system_info['gpu_model'] = gpu_info['gpu_model']
  system_info['gpu_count'] = gpu_info['gpu_count']
  system_info['cpu_model'] = utils.get_cpu_name()
  system_info['cpu_core_count'] = utils.get_cpu_core_count()
  system_info['cpu_socket_count'] = utils.get_cpu_socket_count()

  execution_summary = {}
  execution_summary['execution_id'] = execution_id
  execution_summary['benchmark_result'] = benchmark_result
  execution_summary['benchmark_info'] = benchmark_info
  execution_summary['ml_framework_info'] = ml_framework_info
  execution_summary['system_info'] = system_info

  return execution_summary
