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

"""Collect profiler data for Tensorboard with a separate thread."""

from __future__ import print_function

import logging
import os
import sched
import threading
import time
import traceback

import perfzero.utils as utils


def _start_profiler(output_dir):
  """Start profiler.

  Args:
    output_dir: log directory to place the profiler data
  """
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  profiler_data_dir = os.path.join(output_dir, 'profiler_data')
  utils.make_dir_if_not_exist(profiler_data_dir)
  logging.info('Starting TensorFlow profiler and saving data to dir %s',
                 profiler_data_dir)
  try:
    tf.profiler.experimental.start(profiler_data_dir)
    logging.info('Started TensorFlow profiler')
  except Exception:  # pylint: disable=broad-except
    logging.error('TensorFlow profiler failed to start due to error:\n %s',
                  traceback.format_exc())


def _stop_profiler():
  """Stop profiler."""

  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  try:
    tf.profiler.experimental.stop()
    logging.info('Stopped TensorFlow profiler.')
  except Exception:  # pylint: disable=broad-except
    logging.error('TensorFlow profiler failed to stop due to error:\n %s',
                  traceback.format_exc())


class TensorFlowProfiler(object):
  """Collect profiler data for Tensorboard with a separate thread."""

  def __init__(self, profiler_enabled_time_str, output_dir):
    """Constructor.

    Args:
      profiler_enabled_time_str: the value of the config --profiler_enabled_time
      output_dir: log directory to place the profiler data
    """

    self.profiler_enabled_time_str = profiler_enabled_time_str
    self.output_dir = output_dir
    self.exit_event = threading.Event()
    self.scheduler = sched.scheduler(time.time, self._sleep_until_exit)

  def _sleep_until_exit(self, timeout):
    start_time = time.time()
    cur_time = time.time()
    while cur_time - start_time < timeout and not self.exit_event.is_set():
      time.sleep(min(1, timeout + start_time - cur_time))
      cur_time = time.time()

  def start(self):
    """Schedule start/stop profiler event specified in profiler_enabled_time_str."""

    if not self.profiler_enabled_time_str:
      return

    last_end_time = -1
    for time_str in self.profiler_enabled_time_str.split(','):
      begin_time = int(time_str.split(':')[0].strip())
      end_time_str = time_str.split(':')[1].strip() if ':' in time_str else None
      end_time = int(end_time_str) if end_time_str else 365 * 24 * 60 * 60
      if begin_time <= last_end_time:
        raise ValueError('begin_time {} is no larger than the last '
                         'end_time {}'.format(begin_time, last_end_time))
      if end_time <= begin_time:
        raise ValueError('end_time {} is no larger than begin_time {}'.format(
            end_time, begin_time))
      # 4th positional arg added to support Python2 for the short-term.
      self.scheduler.enter(begin_time, 1, _start_profiler,
        argument=(self.output_dir,))
      self.scheduler.enter(end_time, 1, _stop_profiler, ())  # pylint: disable=no-value-for-parameter
      last_end_time = end_time

    threading.Thread(target=self.scheduler.run).start()

  def stop(self):
    """Stop scheduler and save profiler data if any event is cancelled."""

    event_canceled = False
    for event in self.scheduler.queue:
      try:
        self.scheduler.cancel(event)
        event_canceled = True
      except ValueError:
        # This is OK because the event may have been just canceled
        pass

    # Signal the scheduler thread to stop sleeping
    self.exit_event.set()

    # Save the profiler data if any event is canceled
    if event_canceled:
      _stop_profiler()
