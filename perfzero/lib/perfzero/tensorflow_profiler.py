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


def _start_profiler():
  from tensorflow.python.eager import profiler  # pylint: disable=g-import-not-at-top

  try:
    profiler.start()
    logging.info('Started TensorFlow profiler')
  except Exception:  # pylint: disable=broad-except
    logging.error('TensorFlow profiler failed to start due to error:\n %s',
                  traceback.format_exc())


def _stop_and_save_profiler(output_dir):
  """Stop profiler and save profiler data.

  Args:
    output_dir: log directory to place the profiler data
  """

  from tensorflow.python.eager import profiler  # pylint: disable=g-import-not-at-top

  try:
    profiler_data_dir = os.path.join(output_dir, 'profiler_data')
    logging.info('Stopping TensorFlow profiler and saving data to dir %s',
                 profiler_data_dir)
    utils.make_dir_if_not_exist(profiler_data_dir)
    result = profiler.stop()
    with open(os.path.join(profiler_data_dir, 'local.trace'), 'wb') as f:
      f.write(result)
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
      self.scheduler.enter(begin_time, 1, _start_profiler)  # pylint: disable=no-value-for-parameter
      self.scheduler.enter(end_time, 1, _stop_and_save_profiler,
                           argument=(self.output_dir,))
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
      _stop_and_save_profiler(self.output_dir)

