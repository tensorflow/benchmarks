# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Flask application file for benchmark dashboard."""
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
import json
import logging
from operator import itemgetter
import re
import urllib

from flask import Flask, render_template, request
from google.cloud import datastore


app = Flask(__name__)

# How much data to fetch for graphing.
_DAYS_TO_FETCH = 90
# Don't show a benchmark in benchmark list if it hasn't been run
# for this many days.
_MAX_DAYS_WITHOUT_RUN = 14
# Arguments in this list will not be displayed on the dashboard.
_ARGUMENTS_TO_EXCLUDE = set(
    ['job_name', 'result_storage', 'task_index'])


def argument_name(argument):
  """Gets argument name from string in the form "--arg_name=value".

  Args:
    argument: String argument in the form --arg_name=value.

  Returns:
    String argument name.
  """
  if len(argument) < 4 or argument[:2] != '--' or '=' not in argument:
    logging.error('Invalid argument: %s. Argument must be in the form '
                  '--name=value', argument)
    return ''
  return argument[2:argument.index('=')]


@app.route('/')
@app.route('/<pattern>')
def index(pattern=None):
  """Renders index.html page with a list of benchmarks."""
  filter_regex = None
  if pattern:
    try:
      filter_regex = re.compile(urllib.parse.unquote(pattern))
    except re.error:
      logging.error('Invalid regex.')
      return render_template('index.html', tests=[])

  min_time_to_lookup = datetime.now() - timedelta(days=_MAX_DAYS_WITHOUT_RUN)

  client = datastore.Client()
  query = client.query(kind='Test')
  query.add_filter('start', '>', min_time_to_lookup)

  fetched = list(query.fetch())
  test_names = {}  # maps test name to encoded test name
  for fetched_result in fetched:
    if fetched_result['test'] in test_names:
      continue  # already added
    if not filter_regex or re.search(pattern, fetched_result['test']):
      test_names[fetched_result['test']] = urllib.parse.quote(
          fetched_result['test'], safe='')

  # convert test_names to list and sort
  test_names = sorted(test_names.items(), key=itemgetter(1), reverse=True)

  return render_template('index.html', tests=test_names)


@app.route('/test/<test_id>')
def test(test_id):
  """Renders test.html page with a graph for each benchmark entry."""
  test_id = urllib.parse.unquote(test_id)
  min_time_to_lookup = datetime.now() - timedelta(days=2)
  client = datastore.Client()

  # Get most recent start time for this test
  query = client.query(kind='Test')
  query.add_filter('test', '=', test_id)
  query.order = ['-start']
  test_results = list(query.fetch(limit=1))
  if not test_results:
    return 'No data for benchmark %s' % test_id

  start_time = test_results[0]['start']

  # Get a list of entry ids
  query = client.query(kind='Entry')
  query.add_filter('test', '=', test_id)
  query.add_filter('start', '=', start_time)

  try:
    test_info = json.loads(test_results[0]['info'])
  except ValueError as e:
    logging.exception('Failed to parse "info" in test_results.')
    test_info = None
  arguments = []
  if (test_info and 'runConfiguration' in test_info and
      'argument' in test_info['runConfiguration']):
    arguments = test_info['runConfiguration']['argument']
  arguments = [
      arg for arg in arguments
      if argument_name(arg) not in _ARGUMENTS_TO_EXCLUDE]
  arguments = ' '.join(arguments)
  entries = []
  Entry = namedtuple('Entry', ['id', 'latest_value'])
  for entry in query.fetch():
    info = json.loads(entry['info'])
    entries.append(Entry(entry['entry'], info['wallTime']))

  return render_template(
      'test.html', test_id=test_id, entries=entries,
      latest_time=start_time.strftime('%Y-%m-%d %H:%M'),
      arguments=arguments)


@app.route('/benchmark_data/')
def benchmark_data():
  """Returns benchmark data in json format for graphing."""
  test_id = urllib.parse.unquote(request.args.get('test'))
  entry_id = urllib.parse.unquote(request.args.get('entry'))
  min_time_to_lookup = datetime.now() - timedelta(days=_DAYS_TO_FETCH)

  client = datastore.Client()
  timing_query = client.query(kind='Entry')
  timing_query.add_filter('test', '=', test_id)
  timing_query.add_filter('entry', '=', entry_id)
  timing_query.add_filter('start', '>', min_time_to_lookup)
  timing_query.projection = ['start', 'timing']
  start_and_timing = [
      {'start': data['start'], 'timing': data['timing']}
      for data in timing_query.fetch()]
  start_and_timing_json = json.dumps(start_and_timing)
  return start_and_timing_json


@app.errorhandler(500)
def server_error(e):
  logging.exception('An error occurred during a request.')
  return 'An internal error occurred.', 500


if __name__ == '__main__':
  # This is used when running locally. Gunicorn is used to run the
  # application on Google App Engine. See entrypoint in app.yaml.
  app.run(host='127.0.0.1', port=8080, threaded=True)
