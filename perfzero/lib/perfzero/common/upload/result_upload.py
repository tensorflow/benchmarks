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
from six import u as unicode  # pylint: disable=W0622

import google.auth
from google.cloud import bigquery
from google.cloud.bigquery.dbapi import connect


def upload_result(test_result,
                  result_info,
                  project,
                  dataset='benchmark_results_dev',
                  table='result',
                  test_info=None,
                  system_info=None,
                  extras=None,
                  stream=False):
  """Upload test result.

  Note: Using stream=False has a 1000 per day insert limit per table. Using
  stream=True, the documented limit is 50K+. With streaming there can be
  a small and possibly not noticeable delay to seeing the results the BigQuery
  UI, but there can be a 90 minute more or less delay in the results being part
  of exports.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    test_result: `dict` with core info.  Use `result_info.build_test_result`.
    result_info: `dict` with result info.  Use `result_info.build_test_result`.
    project: Project where BigQuery dataset is located.
    dataset: BigQuery dataset to use.
    table: BigQuery table to insert into.
    test_info: `dict` of test info. Use `result_info.build_test_info`.
    system_info: `dict` of system info. Use `result_info.build_system_info`.
    extras: `dict` of values that will be serialized to JSON.
    stream: Set to true to stream rows.
  """
  if project == 'LOCAL':
    credentials = {}
    row = _build_row(credentials, test_result, result_info, test_info,
                     system_info, extras)
    print('row:{}', row)
  else:
    credentials, _ = google.auth.default()

    row = _build_row(credentials, test_result, result_info, test_info,
                     system_info, extras)

    client = bigquery.Client(project=project, credentials=credentials)

    if stream:
      _stream_upload(client, dataset, table, row)
    else:
      _upload(client, dataset, table, row)


def _upload(client, dataset, table, row):
  """Uploads row to BigQuery."""
  conn = connect(client=client)
  cursor = conn.cursor()
  sql = """INSERT into {}.{} (result_id, test_id, test_harness,
           test_environment, result_info, user, timestamp, system_info,
           test_info, extras)
             VALUES
           (@result_id, @test_id, @test_harness, @test_environment,
           @result_info, @user, @timestamp, @system_info, @test_info, @extras)
           """.format(dataset, table)

  cursor.execute(sql, parameters=row)
  conn.commit()
  # Cursor and connection close on their own as well.
  cursor.close()
  conn.close()


def _stream_upload(client, dataset, table, row):
  """Uploads row to BigQuery via streaming interface."""
  table_ref = client.dataset(dataset).table(table)
  table_obj = client.get_table(table_ref)  # API request
  errors = client.insert_rows(table_obj, [row])
  if errors:
    print('Error inserting rows:{}'.format(errors))


def _build_row(credentials,
               test_result,
               result_info,
               test_info=None,
               system_info=None,
               extras=None):
  """Builds row to be inserted into BigQuery.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    credentials: Result of the test, strongly suggest use Result.
    test_result: `dict` with core info.  Use `result_info.build_test_result`.
    result_info: `dict` with result info.  Use `result_info.build_test_result`.
    test_info: `dict` of test info. Use `result_info.build_test_info`.
    system_info: `dict` of system info. Use `result_info.build_system_info`.
    extras: `dict` of values that will be serialized to JSON.

  Returns:
    `dict` to be inserted into BigQuery.
  """
  row = copy.copy(test_result)
  row['result_id'] = unicode(str(uuid.uuid4()))
  # The user is set to the email address of the service account.  If that is not
  # found, then the logged in user is used as a last best guess.
  if hasattr(credentials, 'service_account_email'):
    row['user'] = credentials.service_account_email
  else:
    row['user'] = unicode(pwd.getpwuid(os.getuid())[0])

  # gpylint warning suggests using a different lib that does not look helpful.
  # pylint: disable=W6421
  row['timestamp'] = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

  # BigQuery expects unicode object and maps that to datatype.STRING.
  row['result_info'] = unicode(json.dumps(result_info))
  row['system_info'] = unicode(json.dumps(system_info if system_info else None))
  row['test_info'] = unicode(json.dumps(test_info) if test_info else None)
  row['extras'] = unicode(json.dumps(extras))

  return row
