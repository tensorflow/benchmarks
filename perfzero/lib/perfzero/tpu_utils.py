"""Utility to manage the tpu version before starting the benchmark."""

import json

from absl import app
from absl import flags
from absl import logging
from six.moves.urllib import request

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  print(
      'Falling back to TensorFlow client; we recommended you install the Cloud '
      'TPU client directly with pip install cloud-tpu-client.')
  from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top

flags.DEFINE_string('started_tpu_address', None,
                    'The TPU address after the tpu was started.')
flags.DEFINE_string('update_tpu_vm_version', None,
                    'The tpu vm version to update the current tpu.')
FLAGS = flags.FLAGS


def _as_text(s):
  """Converts a byte/string into string."""
  if isinstance(s, bytes):
    return s.decode('utf-8')
  return s


def _get_content(url):
  """Opens the url and loads the response into json."""
  logging.info('opening url %s', url)
  req = request.Request(url, data=b'')
  resp = request.urlopen(req)
  resp_text = _as_text(resp.read())
  logging.info('resp_text = %s', resp_text)
  json_data = json.loads(resp_text)
  logging.info('json_data = %s', json_data)
  return json_data


def get_tpu_version(tpu_address):
  """Returns the current software version on tpu."""
  tpu_client = client.Client(tpu=tpu_address)
  tpu_client.wait_for_healthy()
  runtime_version = tpu_client.runtime_version()
  workers = tpu_client.network_endpoints()
  if workers:
    ip_addr = workers[0]['ipAddress']
    url = 'http://{}:8475/requestversion'.format(ip_addr)
    return _get_content(url)
  else:
    logging.error('No tpu endpoint info')
    return None


def main(unused_argv):
  tpu_address = FLAGS.started_tpu_address
  new_vm_id = FLAGS.update_tpu_vm_version
  logging.info('tpu_address: %s, update request: %s', tpu_address, new_vm_id)

  tpu_client = client.Client(tpu=tpu_address)
  logging.info('Waiting for initial tpu healthy.')
  tpu_client.wait_for_healthy()
  logging.info('Initial tpu healthy.')

  if new_vm_id:
    logging.info('Trying configure_tpu_version(%s).', new_vm_id)
    tpu_client.configure_tpu_version(version=new_vm_id)
    logging.info('Waiting for new healthy.')
    tpu_client.wait_for_healthy()
    logging.info('Health check done.')


if __name__ == '__main__':
  app.run(main)
