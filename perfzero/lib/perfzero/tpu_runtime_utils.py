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

def _get_version_info(url):
  """Constructs a version info from the response."""
  json_data = _get_content(url)
  logging.info('json_data = %s', json_data)
  info = {
    'url': '',
    'hash': json_data.get('buildLabel', ''),
    'branch': '',
    'piper_id': json_data.get('piperOriginRevId', '')
  }
  return info

def get_tpu_version(tpu_address):
  """Returns the current software version on tpu."""
  logging.info('Trying to connect to tpu %s', tpu_address)
  tpu_client = client.Client(tpu=tpu_address)
  tpu_client.wait_for_healthy()
  workers = tpu_client.network_endpoints()
  if workers:
    ip_addr = workers[0]['ipAddress']
    url = 'http://{}:8475/requestversion'.format(ip_addr)
    return _get_version_info(url)
  else:
    logging.error('No tpu endpoint info')
    return {
      'url': '',
      'hash': '',
      'branch': '',
      'piper_id': '',
    }
  
def configure_tpu(tpu_params):
  return get_tpu_version(tpu_params.get('name'))
