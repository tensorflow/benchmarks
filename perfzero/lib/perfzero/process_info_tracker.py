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

"""Keep track of process information such as maximum memory usage with a separate thread."""

from __future__ import absolute_import

import json
import logging
import os
import sched
import threading
import time
import traceback
import psutil


class ProcessInfoTracker(object):
  """Keep track of process information such as maximum memory usage with separate thread."""

  def __init__(self, output_dir):
    self.process_info_log = open(os.path.join(output_dir, 'process_info.log'),
                                 'w')
    self.scheduler = sched.scheduler(time.time, time.sleep)
    self.process_info = {}
    self.process_info['max_rss'] = 0
    self.process_info['max_vms'] = 0
    self.process_info['max_cpu_percent'] = 0
    self.exit_event = threading.Event()
    self.last_exception = None
    self.start_time = None

  def start(self):
    self.start_time = time.time()
    self.scheduler.enter(1, 1, self._update_process_info)  # pylint: disable=no-value-for-parameter
    threading.Thread(target=self.scheduler.run).start()
    logging.info('Started process information tracker.')

  def stop(self):
    self.exit_event.set()
    self.process_info_log.flush()
    logging.info('Stopped process information tracker.')

    if self.last_exception is not None:
      raise self.last_exception  # pylint: disable=raising-bad-type

    return dict(self.process_info)

  def _update_process_info(self):
    """Read and update process info using background thread every 1 second."""
    try:
      p = psutil.Process(os.getpid())
      memory_info = p.memory_info()
      # This is a blocking call which takes 0.1 second.
      # This affects the interval # at which the metrics are reported
      cpu_percent = p.cpu_percent(interval=0.1)

      self.process_info['max_rss'] = max(self.process_info['max_rss'],
                                         memory_info.rss)
      self.process_info['max_vms'] = max(self.process_info['max_vms'],
                                         memory_info.vms)
      self.process_info['max_cpu_percent'] = max(
          self.process_info['max_cpu_percent'], cpu_percent)

      entry = {}
      entry['time'] = time.time() - self.start_time
      entry['rss'] = memory_info.rss
      entry['vms'] = memory_info.vms
      entry['cpu_percent'] = cpu_percent
      self.process_info_log.write(json.dumps(entry) + '\n')
      if not self.exit_event.is_set():
        # Schedule the next event to be run after 1 second
        self.scheduler.enter(1, 1, self._update_process_info)  # pylint: disable=no-value-for-parameter
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Process info tracker failed due to error:\n %s',
                    traceback.format_exc())
      self.last_exception = e


