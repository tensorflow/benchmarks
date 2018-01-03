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

"""Utilities for CNN benchmarks."""
from __future__ import print_function

from collections import namedtuple
import sys
import threading

from absl import flags
import tensorflow as tf
flags.DEFINE_boolean('use_python32_barrier', False,
                     """When on, use threading.Barrier at python 3.2.""")
flags.DEFINE_boolean('flush_stdout', False,
                     """When on, flush stdout everytime log_fn is called.""")
FLAGS = flags.FLAGS


def tensorflow_version_tuple():
  v = tf.__version__
  major, minor, patch = v.split('.')
  return (int(major), int(minor), patch)


def tensorflow_version():
  vt = tensorflow_version_tuple()
  return vt[0] * 1000 + vt[1]


def log_fn(log):
  print(log)
  if FLAGS.flush_stdout:
    sys.stdout.flush()

# ParamSpec describes one of benchmark_cnn.BenchmarkCNN's parameters.
ParamSpec = namedtuple('_ParamSpec',
                       ['flag_type', 'default_value', 'description'])


# For Python 2.7 compatibility, we do not use threading.Barrier.
class Barrier(object):
  """Implements a lightweight Barrier.

  Useful for synchronizing a fixed number of threads at known synchronization
  points.  Threads block on 'wait()' and simultaneously return once they have
  all made that call.

  # Implementation adopted from boost/thread/barrier.hpp
  """

  def __init__(self, parties):
    """Create a barrier, initialised to 'parties' threads."""
    self.cond = threading.Condition(threading.Lock())
    self.parties = parties
    # Indicates the number of waiting parties.
    self.waiting = 0
    # generation is needed to deal with spurious wakeups. If self.cond.wait()
    # wakes up for other reasons, generation will force it go back to wait().
    self.generation = 0
    self.broken = False

  def wait(self):
    """Wait for the barrier."""
    with self.cond:
      # Check if the barrier has been disabled or not.
      if self.broken:
        return
      gen = self.generation
      self.waiting += 1
      if self.waiting == self.parties:
        self.waiting = 0
        self.generation += 1
        self.cond.notify_all()
      # loop because of spurious wakeups
      while gen == self.generation:
        self.cond.wait()

  # TODO(huangyp): Remove this method once we find a way to know which step
  # is the last barrier.
  def abort(self):
    """Clear existing barrier and disable this barrier."""
    with self.cond:
      if self.waiting > 0:
        self.generation += 1
        self.cond.notify_all()
      self.broken = True


class ImageProducer(object):
  """An image producer that puts images into a staging area periodically.

  This class is useful for periodically running a set of ops, `put_ops` on a
  different thread every `batch_group_size` steps.

  The notify_image_consumption() method is used to increment an internal counter
  so that every `batch_group_size` times it is called, `put_ops` is executed. A
  barrier is placed so that notify_image_consumption() will block until
  the previous call to `put_ops` has been executed.

  The start() method is used to start the thread that runs `put_ops`.

  The done() method waits until the last put_ops is executed and stops the
  thread.

  The purpose of this class is to fill an image input pipeline every
  `batch_group_size` steps. Suppose `put_ops` supplies `batch_group_size` images
  to the input pipeline when run, and that every step, 1 batch of images is
  consumed. Then, by calling notify_image_consumption() every step, images are
  supplied to the input pipeline at the same amount they are consumed.

  Example usage:
  ```
  put_ops = ... # Enqueues `batch_group_size` batches to a StagingArea
  get_op = ...  # Dequeues 1 batch, and does some operations on it
  batch_group_size = 4
  with tf.Session() as sess:
    image_producer = cnn_util.ImageProducer(sess, put_op, batch_group_size)
    image_producer.start()
    for _ in range(100):
      sess.run(get_op)
      image_producer.notify_image_consumption()
  ```
  """

  def __init__(self, sess, put_ops, batch_group_size):
    self.sess = sess
    self.num_gets = 0
    self.put_ops = put_ops
    self.batch_group_size = batch_group_size
    self.done_event = threading.Event()
    if (FLAGS.use_python32_barrier and
        sys.version_info[0] == 3 and sys.version_info[1] >= 2):
      self.put_barrier = threading.Barrier(2)
    else:
      self.put_barrier = Barrier(2)

  def _should_put(self):
    return (self.num_gets + 1) % self.batch_group_size == 0

  def done(self):
    """Stop the image producer."""
    self.done_event.set()
    self.put_barrier.abort()
    self.thread.join()

  def start(self):
    """Start the image producer."""
    self.sess.run([self.put_ops])
    self.thread = threading.Thread(target=self._loop_producer)
    # Set daemon to true to allow Ctrl + C to terminate all threads.
    self.thread.daemon = True
    self.thread.start()

  def notify_image_consumption(self):
    """Increment the counter of image_producer by 1.

    This should only be called by the main thread that consumes images and runs
    the model computation. One batch of images should be consumed between
    calling start() and the first call to this method. Then, one batch of images
    should be consumed between any two successive calls to this method.
    """
    if self._should_put():
      self.put_barrier.wait()
    self.num_gets += 1

  def _loop_producer(self):
    while not self.done_event.isSet():
      self.sess.run([self.put_ops])
      self.put_barrier.wait()


class BaseClusterManager(object):
  """The manager for the cluster of servers running the benchmark."""

  def __init__(self, params):
    worker_hosts = params.worker_hosts.split(',')
    ps_hosts = params.ps_hosts.split(',') if params.ps_hosts else []
    cluster = {'worker': worker_hosts}
    if ps_hosts:
      cluster['ps'] = ps_hosts
    self._cluster_spec = tf.train.ClusterSpec(cluster)

  def get_target(self):
    """Returns a target to be passed to tf.Session()."""
    raise NotImplementedError('get_target must be implemented by subclass')

  def join_server(self):
    raise NotImplementedError('join must be implemented by subclass')

  def get_cluster_spec(self):
    return self._cluster_spec

  def num_workers(self):
    return len(self._cluster_spec.job_tasks('worker'))

  def num_ps(self):
    if 'ps' in self._cluster_spec.jobs:
      return len(self._cluster_spec.job_tasks('ps'))
    else:
      return 0


class GrpcClusterManager(BaseClusterManager):
  """A cluster manager for a cluster networked with gRPC."""

  def __init__(self, params, config_proto):
    super(GrpcClusterManager, self).__init__(params)
    if params.job_name == 'controller':
      self._target = 'grpc://%s' % self._cluster_spec.job_tasks('worker')[0]
    else:
      self._server = tf.train.Server(self._cluster_spec,
                                     job_name=params.job_name,
                                     task_index=params.task_index,
                                     config=config_proto,
                                     protocol=params.server_protocol)
      self._target = self._server.target

  def get_target(self):
    return self._target

  def join_server(self):
    return self._server.join()
