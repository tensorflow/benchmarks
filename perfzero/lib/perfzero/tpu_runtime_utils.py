"""Utility to manage the tpu version before starting the benchmark."""

import json
from absl import logging

from six.moves.urllib import request

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  print(
      'Falling back to TensorFlow client; we recommended you install the Cloud '
      'TPU client directly with pip install cloud-tpu-client.')
  from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top


def _as_text(s):
  """Converts a byte/string into string."""
  if isinstance(s, bytes):
    return s.decode('utf-8')
  return s


def _get_content(url):
  """Opens the url and loads the response into json."""
  logging.info('opening url %s', url)
  req = request.Request(url)
  resp = request.urlopen(req)
  resp_text = _as_text(resp.read())
  logging.info('response text = %s', resp_text)
  return json.loads(resp_text)


def _get_version_info(url, version_label):
  """Constructs a version info from the response."""
  json_data = _get_content(url)
  logging.info('json_data = %s', json_data)
  if 'currentVersion' in json_data:
    commit_id = json_data['currentVersion']
  elif 'buildLabel' in json_data:
    commit_id = json_data['buildLabel']
  else:
    commit_id = ''
    
  info = {
      'url': '',
      'hash': commit_id,
      'branch': version_label,
      'piper_id': json_data.get('piperOriginRevId', '')
  }
  return info



def _configure_tpu_version(tpu_name, tpu_zone, tpu_project, version_label, new_version_id):
  """Returns the current tpu version after resetting to an optional version."""
  # The tpu_name is arbitrary / user chosen unique string for this tpu.
  logging.info('Trying to connect to tpu %s', tpu_name)
  tpu_client = client.Client(tpu=tpu_name, zone=tpu_zone, project=tpu_project)
  tpu_client.wait_for_healthy()

  if new_version_id:
    logging.info('Trying to reset tpu version to %s', new_version_id)
    tpu_client.configure_tpu_version(version=new_version_id)
    tpu_client.wait_for_healthy()
    logging.info('TPU healthy after version reset. New version id: %s', new_version_id)
  else:
    logging.info('Using the default or pre-started tpu version id.')

  workers = tpu_client.network_endpoints()
  if workers:
    ip_addr = workers[0]['ipAddress']
    url = 'http://{}:8475/requestversion'.format(ip_addr)
    return _get_version_info(url, version_label)
  else:
    logging.error('No tpu endpoint info')
    return {
        'url': '',
        'hash': '',
        'branch': version_label,
        'piper_id': '',
    }


def configure_tpu(tpu_params):
  return _configure_tpu_version(
      tpu_name=tpu_params.get('name'),
      tpu_zone=tpu_params.get('zone'),
      tpu_project=tpu_params.get('project'),
      version_label=tpu_params.get('version'),
      new_version_id=tpu_params.get('version_id'))
