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

"""Defines VariableMgr and subclasses used to manage variables.

"""

from __future__ import print_function

import collections as pycoll
import operator
import re

import tensorflow as tf

from tensorflow.contrib import nccl
from tensorflow.contrib.all_reduce.python import all_reduce
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops

PS_SHADOW_VAR_PREFIX = 'ps_var'


# To be used with custom_getter on tf.get_variable.
class OverrideCachingDevice(object):
  """Variable getter which caches variables on the least loaded device.

  Variables smaller than a certain threshold are cached on a single specific
  device, as specified in the constructor. All other variables are load balanced
  across a pool of devices, by caching each variable on the least loaded device.
  """

  def __init__(self, devices, device_for_small_variables,
               small_variable_size_threshold):
    self.devices = devices
    self.sizes = [0] * len(self.devices)
    self.device_for_small_variables = device_for_small_variables
    self.small_variable_size_threshold = small_variable_size_threshold

  def __call__(self, getter, *args, **kwargs):
    size = tf.TensorShape(kwargs['shape']).num_elements()
    if size < self.small_variable_size_threshold:
      device_name = self.device_for_small_variables
    else:
      device_index, _ = min(enumerate(
          self.sizes), key=operator.itemgetter(1))
      device_name = self.devices[device_index]
      self.sizes[device_index] += size

    kwargs['caching_device'] = device_name
    var = getter(*args, **kwargs)
    return var


# To be used with custom_getter on tf.get_variable. Ensures the created variable
# is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
class OverrideToLocalVariableIfNotPsVar(object):

  # args and kwargs come from the custom_getter interface for Tensorflow
  # variables, and matches tf.get_variable's signature, with the addition of
  # 'getter' at the beginning.
  def __call__(self, getter, name, *args, **kwargs):
    if name.startswith(PS_SHADOW_VAR_PREFIX):
      return getter(*args, **kwargs)

    if 'collections' in kwargs:
      collections = kwargs['collections']
    if not collections:
      collections = set([tf.GraphKeys.GLOBAL_VARIABLES])
    else:
      collections = set(collections.copy())
    collections.remove(tf.GraphKeys.GLOBAL_VARIABLES)
    collections.add(tf.GraphKeys.LOCAL_VARIABLES)
    kwargs['collections'] = list(collections)
    return getter(name, *args, **kwargs)


class ParamServerDeviceSetter(object):
  """Helper class to assign variables on the least loaded ps-device."""

  def __init__(self, worker_device, ps_devices):
    """Initializer for ParamServerDevicSetter.

    Args:
      worker_device: the device to use for computer ops.
      ps_devices: a list of device to use for Variable ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2']:
      return self.worker_device

    device_index, _ = min(enumerate(
        self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCnn to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """

  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []

  def each_tower_has_variables(self):
    """Returns True if each GPU tower of the model has separate variables."""
    assert False, 'Must be implemented in subclass'

  def supports_staged_vars(self):
    """Whether staged variable management is supported."""
    return False

  def create_outer_variable_scope(self, device_num):
    """Create the tf.variable_scope around all model graph operations."""
    del device_num  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def preprocess_device_grads(self, device_grads):
    """Preprocess the device gradients prior to applying them.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns: a tuple of (apply_gradients_devices, gradient_state).
      gradient_state is an opaque structure that should be passed to
      get_gradients_to_apply() and append_apply_gradients_ops() (in that order).
      apply_gradients_devices is a list of devices where the gradients will be
      applied with get_gradients_to_apply() and append_apply_gradients_ops().
    """
    del device_grads  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_gradients_to_apply(self, device_num, gradient_state):
    """Returns the [(gradient, variable)] list to apply for device_num.

    Args:
      device_num: indexes into apply_gradients_devices, which was returned by an
        earlier call to preprocess_device_grads.
      gradient_state: from previous call to apply_gradients_devices.
    """
    del device_num, gradient_state  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def append_apply_gradients_ops(
      self, gradient_state, opt, grads, training_ops):
    """Adds training ops for grads to 'training_ops'.

    Args:
      gradient_state: from previous call to apply_gradients_devices.
      opt: the underlying optimizer
      grads: [(grad, var)] to apply
      training_ops: list to which to add ops
    """
    del gradient_state  # unused by this implementation
    apply_gradients_op = opt.apply_gradients(grads)
    training_ops.append(apply_gradients_op)

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self, rel_device_num, abs_device_num,
                                    writable=False):
    """Return the set of trainable variables on device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable vairalbes on the specified device.
    """
    del rel_device_num, writable
    if self.each_tower_has_variables():
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('v%s/' % abs_device_num)
      ]
    else:
      params = tf.trainable_variables()
    return params


class VariableMgrIndependent(VariableMgr):
  """VariableMgr that implements the --independent mode for local jobs.

     Each GPU has its own copy of the variables, and gradients are
     not shared between towers. This can be used to check
     performance when no data is moved between GPUs.
  """

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num)

  def preprocess_device_grads(self, device_grads):
    return (self.benchmark_cnn.devices, device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    return device_grads[device_num]

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrLocalFetchFromPS(VariableMgr):
  """VariableMgr that implements the --parameter_server mode for local jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v', reuse=bool(device_num))

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    device_grads = gradient_state
    return aggregate_gradients_using_copy_with_variable_colocation(
        device_grads, use_mean=True)

  def get_devices(self):
    raw_devices = self.benchmark_cnn.raw_devices
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      return [ParamServerDeviceSetter(d, raw_devices) for d in raw_devices]
    else:
      return [tf.train.replica_device_setter(
          worker_device=d, ps_device=self.benchmark_cnn.param_server_device,
          ps_tasks=1) for d in raw_devices]


class StagedModelVariable(object):
  """Staging variable wrapper that decouples reads and updates.

  This class represents a variable through a staging buffer. Reads from this
  variable directly gets from the staging buffer. Updates are stacked into
  another staging buffer, and will be processed later.
  """

  def __init__(self, real_var, var_stage_get, variable_mgr):
    """Initializer for the model variables through a staging buffer.

    Args:
      real_var: the underlying real variable.
      var_stage_get: the read op from the staging buffer.
      variable_mgr: the parent variable-manager.
    """
    self.real_var = real_var
    self.var_stage_get = var_stage_get
    self.variable_mgr = variable_mgr

  def _value(self):
    """The read access of this variable. The content from the staging buffer."""
    return self.var_stage_get

  def _ref(self):
    """Return the underlying variable ref, required by tf.colocate_with."""
    return self.real_var._ref()  # pylint: disable=protected-access

  def read_value(self):
    """Mimics tf.Variable.read_value()."""
    return tf.identity(self.var_stage_get, name='read')

  @property
  def dtype(self):
    """Return the non-reference dtype."""
    return self.var_stage_get.dtype

  def assign_sub(self, delta, name=None):
    """Mimic the updates to the variable.

    Args:
      delta: is pushed into a staging buffer and will be pumped later.
      name: currently ignored; names of ops and the StagingArea are
            computed without using this pass name.
    Returns:
      The actual updates. The colocation constraint will be reapplied.
    """
    # This parameter is ignored: the StagingArea only supports setting
    # the shared name, not the names of individual ops it uses.
    del name

    # colocate_with(None, True) clears the colocation constraints.
    # Push the delta into a staging buffer.
    with ops.colocate_with(None, True), tf.device(self.var_stage_get.device):
      delta_staging_area = data_flow_ops.StagingArea(
          [self.var_stage_get.dtype], shapes=[self.var_stage_get.shape])
      delta_put_op = delta_staging_area.put([delta])
      self.variable_mgr.staging_delta_ops.append(delta_put_op)
      delta_get_op = delta_staging_area.get()[0]
    # Return the actual updates. The colocation constraint will be reapplied.
    return self.real_var.assign_sub(delta_get_op)

  @staticmethod
  # pylint: disable=bad-staticmethod-argument,invalid-name
  def _TensorConversionFunction(self, dtype=None, name=None, as_ref=False):
    """Utility function for converting a StagedModelVariable to a Tensor."""
    del dtype, name  # unused: this function returns the cached ref or value.
    if as_ref:
      return self._ref()
    else:
      return self._value()


ops.register_tensor_conversion_function(
    StagedModelVariable, StagedModelVariable._TensorConversionFunction)  # pylint: disable=protected-access


class StagedVariableGetter(object):
  """A variable getter through staging buffers on devices.

  Instead of a caching device, this getter tracks where the variable is used.
  And on each device, it goes through a staging buffer.
  """

  def __init__(self, device_num, devices, cpu_device, variable_mgr):
    """Initializer for StagedVariableGetter.

    Args:
      device_num: the current device index.
      devices: a list of all the devices to build towers.
      cpu_device: a cpu_device for this replica. If None, no cpu-caching is
          done.
      variable_mgr: the parent variable manager.
    """
    self.device_num = device_num
    self.devices = devices
    self.cpu_device = cpu_device
    self.variable_mgr = variable_mgr

  def __call__(self, getter, name, *args, **kwargs):
    staging_ops = self.variable_mgr.staging_vars_on_devices[self.device_num]
    if name in staging_ops:
      put_op, get_op = staging_ops[name]
      return get_op
    real_var = getter(name, *args, **kwargs)
    shape = kwargs['shape']
    dtype = kwargs['dtype']
    trainable = kwargs['trainable']
    if self.cpu_device:
      with tf.device(self.cpu_device):
        # This helps copying the weights from the parameter to this server only
        # once.
        if name in self.variable_mgr.staged_vars_on_cpu:
          cpu_var = self.variable_mgr.staged_vars_on_cpu[name]
        else:
          cpu_var = tf.identity(real_var)
          self.variable_mgr.staged_vars_on_cpu[name] = cpu_var
      var_to_stage = cpu_var
    else:
      var_to_stage = tf.identity(real_var)  # de-reference the variable.

    with tf.device(self.devices[self.device_num]):
      staging_area = data_flow_ops.StagingArea([dtype], shapes=[shape])
      put_op = staging_area.put([var_to_stage])
      get_op = staging_area.get()[0]
      staging_ops[name] = (put_op, get_op)
    if trainable:
      # For trainable variables, they are managed separatedly through
      # apply_gradients.
      return get_op
    else:
      # For other shadow variables, the access is decoupled through a wrapper
      # class.
      return StagedModelVariable(real_var, get_op, self.variable_mgr)

  def trainable_variables_on_device(self, rel_device_num, abs_device_num,
                                    writable):
    """Return the set of trainable variables on the specified device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether the returned variables is writable or read-only.

    Returns:
      Return the set of trainable variables on the specified device.
    """
    del abs_device_num
    params_refs = tf.trainable_variables()
    if writable:
      return params_refs
    params = []
    for param in params_refs:
      var_name = param.name.split(':')[0]
      _, var_get_op = self.variable_mgr.staging_vars_on_devices[rel_device_num][
          var_name]
      params.append(var_get_op)
    return params


class VariableMgrLocalFetchFromStagedPS(VariableMgrLocalFetchFromPS):
  """Implements fetching a local variable through staging buffers.
  """

  def __init__(self, benchmark_cnn):
    super(VariableMgrLocalFetchFromStagedPS, self).__init__(benchmark_cnn)
    # A data structure to track where the variables are used on each device.
    # Indexed by device_num and var_name, each entry stores the "put" and "get"
    # ops used for that variable on that device:
    #   staging_vars_on_devices[device_num][var_name] == (put_op, get_op)
    self.staging_vars_on_devices = [dict() for _ in
                                    self.benchmark_cnn.raw_devices]

  def supports_staged_vars(self):
    return True

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices, None, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=self._custom_getter)

  def trainable_variables_on_device(self, rel_device_num, abs_device_num,
                                    writable=False):
    return self._custom_getter.trainable_variables_on_device(
        rel_device_num, abs_device_num, writable=writable)


AllReduceSpecTuple = pycoll.namedtuple('AllReduceSpecTuple', 'alg shards limit')


def parse_general_int(s):
  """Parse integer with power-of-2 suffix eg. 32k."""
  mo = re.match(r'(\d+)([KkMGT]?)$', s)
  if mo:
    i, suffix = mo.group(1, 2)
    v = int(i)
    if suffix:
      if suffix == 'K' or suffix == 'k':
        v *= 1024
      elif suffix == 'M':
        v *= (1024 * 1024)
      elif suffix == 'G':
        v *= (1024 * 1024 * 1024)
      elif suffix == 'T':
        v *= (1024 * 1024 * 1024 * 1024)
      else:
        raise ValueError('invalid integer string %s' % s)
    return v
  else:
    v = int(s)
  return v


def parse_all_reduce_spec(all_reduce_spec):
  """Parse all_reduce_spec.

  Args:
    all_reduce_spec: a string specifying a combination of all-reduce
      algorithms to apply for gradient reduction.

  Returns:
    a list of AllReduceSpecTuple.

  Raises:
    ValueError: all_reduce_spec is not well-formed.

  An all_reduce_spec has BNF form:
     int ::= positive whole number
     g_int ::= int[KkMGT]?
     alg_spec ::= alg | alg#int
     range_spec ::= alg_spec | alg_spec/alg_spec
     spec ::= range_spec | range_spec:g_int:range_spec

  Not all syntactically correct specifications are supported.
  Examples of supported all_reduce_spec strings, with semantics explained:

    'xring' == apply ring all-reduce to all tensors
    'xring#2' == apply ring all-reduce to all tensors, using two simultaneous
            transfer rings, each operating on 1/2 of each tensor.
    'nccl'  == apply NCCL all-reduce to all tensors (only works within
            a single worker process where all devices are GPUs)
    'nccl/xring' == apply NCCL all-reduce to all tensors within each worker
            to produce at least one full-reduced (locally) value,
            then apply ring all-reduce to one such value from each
            worker, then apply NCCL broadcast to propagate those globally
            reduced values back to every device within each worker.
    'pscpu' == Shuffle reduce using worker CPUs as the gather devices: each
            distributed tensor is reduced by copying all instances to
            one of the worker CPUs, computing the reduction there, then
            copying back to each participating device.  Tensor reductions
            are assigned to specific CPUs round-robin.
    'psgpu#4' == Arrange all GPUs across all workers into groups of 4.
            Each distributed tensor is shuffle reduced against one
            such group of 4 GPUs, selected round-robin.  That is, each
            tensor is split across 4 shards for the reduction.
    'pscpu:2k:pscpu#2:64k:xring' == Apply single-shard pscpu to
            tensors of size <= 2048 elements, apply 2-shard pscpu to
            tensors up to size 64k elements, apply xring to larger tensors.
    'pscpu/pscpu#2' == Use shuffle gather to locally reduce each tensor on
            the worker's CPU, then use 2-shard shuffle to reduce those
            locally reduced tensors across workers (on the worker CPUs), then
            scatter the globally reduced values locally from each worker CPU.
  """
  range_parts = all_reduce_spec.split(':') + ['-1']
  if len(range_parts) % 2:
    raise ValueError('all_reduce_spec not well formed: %s' % all_reduce_spec)
  limit = 0
  spec = []
  alg = None
  shards = 1
  for i, range_part in enumerate(range_parts):
    if i % 2 == 1:
      try:
        limit = parse_general_int(range_part)
        spec.append(AllReduceSpecTuple(alg=alg, shards=shards, limit=limit))
      except ValueError:
        raise ValueError('all_reduce_spec (%s) contains non-integer range %s' %
                         (all_reduce_spec, range_part))
    else:
      alg = range_part
      alg_parts = range_part.split('#')
      alg = alg_parts[0]
      if len(alg_parts) > 1:
        try:
          shards = int(alg_parts[1])
        except ValueError:
          raise ValueError('all_reduce_spec (%s) contains non-integer '
                           'shards %s' % all_reduce_spec, alg_parts[1])
      else:
        shards = 1
      if alg not in['nccl', 'nccl/xring', 'nccl/rechd', 'nccl/pscpu',
                    'xring', 'pscpu', 'psgpu', 'pscpu/pscpu']:
        raise ValueError('all_reduce_spec (%s) contains invalid alg %s' %
                         (all_reduce_spec, alg))
  return spec


def build_all_reduce_device_prefixes(job_name, num_tasks):
  """Build list of device prefix names for all_reduce.

  Args:
    job_name: 'worker', 'ps' or 'localhost'.
    num_tasks: number of jobs across which device names should be generated.

  Returns:
     A list of device name prefix strings. Each element spells out the full
     host name without adding the device.
     e.g. '/job:worker/task:0'
  """
  if job_name != 'localhost':
    return ['/job:%s/task:%d' % (job_name, d) for d in range(0, num_tasks)]
  else:
    assert num_tasks == 1
    return ['/job:%s' % job_name]


def group_device_names(devices, group_size):
  """Group device names into groups of group_size.

  Args:
    devices: list of strings naming devices.
    group_size: int >= 1

  Returns:
    list of lists of devices, where each inner list is group_size long,
      and each device appears at least once in an inner list.  If
      len(devices) % group_size = 0 then each device will appear
      exactly once.

  Raises:
    ValueError: group_size > len(devices)
  """
  num_devices = len(devices)
  if group_size > num_devices:
    raise ValueError('only %d devices, but group_size=%d' % (
        num_devices, group_size))
  num_groups = (num_devices // group_size) + (
      1 if (num_devices % group_size != 0) else 0)
  groups = [[] for i in range(num_groups)]
  for i in range(0, num_groups * group_size):
    groups[i % num_groups].append(devices[i % num_devices])
  return groups


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either a local all-reduce algorithm is applied or a regular
     cross-device aggregation is used to replicate the combined
     gradients to all towers.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec):
    super(VariableMgrLocalReplicated, self).__init__(benchmark_cnn)
    if all_reduce_spec:
      spec = parse_all_reduce_spec(all_reduce_spec)
      if len(spec) != 1:
        raise ValueError(
            'replicated mode does not support hybrid all-reduce strategies')
      self._all_reduce_spec = spec[0]
    else:
      self._all_reduce_spec = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num)

  def preprocess_device_grads(self, device_grads):
    if self._all_reduce_spec:
      aggregated_device_grads = sum_gradients_all_reduce(
          ['/job:localhost'], device_grads, 1,
          self._all_reduce_spec.alg,
          self._all_reduce_spec.shards, self.benchmark_cnn.gpu_indices)
    else:
      agg_grads = aggregate_gradients_using_copy_with_device_selection(
          self.benchmark_cnn, device_grads, use_mean=False)
      aggregated_device_grads = []
      for arr in device_grads:
        aggregated_device_grads.append(
            [(g, v) for (_, v), (g, _) in zip(arr, agg_grads)])
    return (self.benchmark_cnn.devices, aggregated_device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    return device_grads[device_num]

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrDistributedAllReduce(VariableMgr):
  """VariableMgr that implements the --distributed_all_reduce mode.

     Each GPU has its own copy of the variables. To apply gradients,
     the specified all-reduce algorithm is used to reduce the gradients
     and replicate the final value to all GPUs.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec, job_name,
               num_workers):
    super(VariableMgrDistributedAllReduce, self).__init__(benchmark_cnn)
    self._all_reduce_spec = parse_all_reduce_spec(all_reduce_spec)
    self._all_reduce_device_prefixes = build_all_reduce_device_prefixes(
        job_name, num_workers)
    self._num_workers = num_workers
    if not self._all_reduce_spec:
      raise ValueError('all_reduce_spec must be specified')

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    """Create a scope for the named device.

    Args:
      device_num: index of device for variable scope. (Note that
        device_num spans all processes in cluster since a single global
        graph is used.)

    Returns:
      the requested variable_scope
    """
    return tf.variable_scope('v%s' % device_num)

  def preprocess_device_grads(self, device_grads):
    remaining_grads = device_grads
    aggregated_grads = []
    for spec_tuple in self._all_reduce_spec:
      if spec_tuple.limit < 0:
        this_grads = remaining_grads
        remaining_grads = []
      else:
        (this_grads, remaining_grads) = split_grads_by_size(
            spec_tuple.limit, remaining_grads)
      if this_grads:
        range_agg_grads = sum_gradients_all_reduce(
            self._all_reduce_device_prefixes, this_grads, self._num_workers,
            spec_tuple.alg, spec_tuple.shards, self.benchmark_cnn.gpu_indices)
        if not aggregated_grads:
          aggregated_grads = range_agg_grads
        else:
          assert len(aggregated_grads) == len(range_agg_grads)
          for i in range(len(aggregated_grads)):
            aggregated_grads[i] += range_agg_grads[i]
    assert not remaining_grads
    full_device_set = []
    for grads in device_grads:
      g, v = grads[0]
      del v
      full_device_set.append(g.device)
    return (full_device_set, aggregated_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    if device_num >= len(device_grads):
      raise ValueError('device_num %d exceeds length of device_grads (%d)' %
                       (device_num, len(device_grads)))
    return device_grads[device_num]

  def get_post_init_ops(self):
    """Copy initialized values for variables to other devices."""
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrDistributedFetchFromPS(VariableMgr):
  """Implements --variable_update=parameter_server mode for distributed jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      caching_devices = self.benchmark_cnn.raw_devices
    else:
      caching_devices = [self.benchmark_cnn.cpu_device]
    custom_getter = OverrideCachingDevice(
        caching_devices, self.benchmark_cnn.cpu_device, 1024*64)
    return tf.variable_scope('v', reuse=bool(device_num),
                             custom_getter=custom_getter)

  def preprocess_device_grads(self, device_grads):
    # Returns (gradient_devices, gradient_state)
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    return aggregate_gradients_using_copy(gradient_state, use_mean=True)

  def get_devices(self):
    ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
        len(self.benchmark_cnn.ps_hosts), tf.contrib.training.byte_size_load_fn)
    return [tf.train.replica_device_setter(
        worker_device=d, cluster=self.benchmark_cnn.cluster,
        ps_strategy=ps_strategy)
            for d in self.benchmark_cnn.raw_devices]


class VariableMgrDistributedFetchFromStagedPS(
    VariableMgrDistributedFetchFromPS):
  """Extends VariableMgrDistributedFetchFromPS for --staged_vars."""

  def __init__(self, benchmark_cnn):
    super(VariableMgrDistributedFetchFromStagedPS, self).__init__(benchmark_cnn)
    self.staging_vars_on_devices = [dict() for _ in
                                    self.benchmark_cnn.raw_devices]
    self.staged_vars_on_cpu = {}

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices,
        self.benchmark_cnn.cpu_device, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=self._custom_getter)

  def supports_staged_vars(self):
    return True

  def trainable_variables_on_device(self, rel_device_num, abs_device_num,
                                    writable=False):
    return self._custom_getter.trainable_variables_on_device(
        rel_device_num, abs_device_num, writable=writable)


class VariableMgrDistributedReplicated(VariableMgr):
  """VariableMgr that implements the --distributed_replicated mode.

     Each GPU has a copy of the variables, and updates its copy after the
     parameter servers are all updated with the gradients from all servers. Only
     works with cross_replica_sync=true. Unlike 'replicated', does not use nccl
     all-reduce for replicating within a server.
  """

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope(
        'v%s' % device_num,
        custom_getter=OverrideToLocalVariableIfNotPsVar())

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    avg_grads = aggregate_gradients_using_copy_with_device_selection(
        self.benchmark_cnn, device_grads, use_mean=True)

    # Make shadow variable on a parameter server for each original trainable
    # variable.
    for i, (g, v) in enumerate(avg_grads):
      my_name = PS_SHADOW_VAR_PREFIX + '/' + v.name
      if my_name.endswith(':0'): my_name = my_name[:-2]
      new_v = tf.get_variable(my_name, dtype=v.dtype.base_dtype,
                              initializer=v.initial_value,
                              trainable=True)
      avg_grads[i] = (g, new_v)
    return avg_grads

  def append_apply_gradients_ops(self, gradient_state, opt,
                                 grads, training_ops):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    # For each variable, apply the combined gradients for this server on
    # the parameter server, and then wait for all other servers to do
    # this.
    for i, (g, v) in enumerate(grads):
      apply_gradient_op = opt.apply_gradients([(g, v)])
      barrier = self.benchmark_cnn.add_sync_queues_and_barrier(
          'replicate_variable_%s' % i, [apply_gradient_op])
      with tf.control_dependencies([barrier]):
        with tf.device(self.benchmark_cnn.cpu_device):
          updated_value = v.read_value()
          for my_d in range(len(self.benchmark_cnn.devices)):
            training_ops.append(
                device_grads[my_d][i][1].assign(updated_value))

  def _strip_port(self, s):
    if s.endswith(':0'):
      return s[:-2]
    return s

  def get_post_init_ops(self):
    # Copy initialized variables for variables on the parameter server
    # to the local copy of the variable.

    local_vars = tf.local_variables()
    local_var_by_name = dict(
        [(self._strip_port(v.name), v) for v in local_vars])
    post_init_ops = []
    for v in tf.global_variables():
      if v.name.startswith(PS_SHADOW_VAR_PREFIX + '/v0/'):
        prefix = self._strip_port(
            v.name[len(PS_SHADOW_VAR_PREFIX + '/v0'):])
        for i in range(self.benchmark_cnn.num_gpus):
          name = 'v%s%s' % (i, prefix)
          if name in local_var_by_name:
            copy_to = local_var_by_name[name]
            post_init_ops.append(copy_to.assign(v.read_value()))
    return post_init_ops

  def _remove_shadow_var_prefix_if_present(self, var_name):
    if var_name.startswith(PS_SHADOW_VAR_PREFIX + '/'):
      return var_name[len(PS_SHADOW_VAR_PREFIX + '/'):]
    else:
      return var_name

  def var_dict_name(self, v):
    return self._strip_port(self._remove_shadow_var_prefix_if_present(v.name))

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    params = {}
    for v in tf.global_variables():
      assert (v.name.startswith(PS_SHADOW_VAR_PREFIX + '/v0/') or
              v.name == 'global_step:0')
      # We store variables in the checkpoint with the shadow variable prefix
      # removed so we can evaluate checkpoints in non-distributed replicated
      # mode. The checkpoints can also be loaded for training in
      # distributed_replicated mode.
      name = self._strip_port(self._remove_shadow_var_prefix_if_present(v.name))
      params[name] = v
    for v in tf.local_variables():
      # Non-trainable variables, such as batch norm moving averages, do not have
      # corresponding global shadow variables, so we add them here. Trainable
      # local variables have corresponding global shadow variables, which were
      # added in the global variable loop above.
      if v.name.startswith('v0/') and v not in tf.trainable_variables():
        params[self._strip_port(v.name)] = v
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


def split_grads_by_size(threshold_size, device_grads):
  """Break gradients into two sets according to tensor size.

  Args:
    threshold_size: int size cutoff for small vs large tensor.
    device_grads: List of lists of (gradient, variable) tuples.  The outer
        list is over devices, the inner list is over individual gradients.

  Returns:
    small_grads: Subset of device_grads where shape is <= theshold_size
       elements.
    large_grads: Subset of device_grads where shape is > threshold_size
       elements.
  """
  small_grads = []
  large_grads = []
  for dl in device_grads:
    small_dl = []
    large_dl = []
    for (g, v) in dl:
      tensor_size = g.get_shape().num_elements()
      if tensor_size <= threshold_size:
        small_dl.append([g, v])
      else:
        large_dl.append([g, v])
    if small_dl:
      small_grads.append(small_dl)
    if large_dl:
      large_grads.append(large_dl)
  return small_grads, large_grads


def sum_grad_and_var_all_reduce(grad_and_vars, num_workers, alg, gpu_indices,
                                aux_devices=None, num_shards=1):
  """Apply all-reduce algorithm over specified gradient tensors."""
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
  scaled_grads = [g for g, _ in grad_and_vars]
  if alg == 'nccl':
    summed_grads = nccl.all_sum(scaled_grads)
  elif alg == 'xring':
    summed_grads = all_reduce.build_ring_all_reduce(
        scaled_grads, num_workers, num_shards, gpu_indices, tf.add)
  elif alg == 'nccl/xring':
    summed_grads = all_reduce.build_nccl_then_ring(scaled_grads, num_shards,
                                                   tf.add)
  elif alg == 'nccl/rechd':
    summed_grads = all_reduce.build_nccl_then_recursive_hd(scaled_grads, tf.add)
  elif alg == 'nccl/pscpu':
    summed_grads = all_reduce.build_nccl_then_shuffle(
        scaled_grads, aux_devices, tf.add, tf.add_n)
  elif alg == 'pscpu/pscpu':
    summed_grads = all_reduce.build_shuffle_then_shuffle(
        scaled_grads, aux_devices,
        # TODO(tucker): devise a way of better specifying the device set
        # for the second level.
        [aux_devices[0]],
        tf.add_n)
  elif alg in ['pscpu', 'psgpu']:
    summed_grads = all_reduce.build_shuffle_all_reduce(
        scaled_grads, aux_devices, tf.add_n)
  else:
    raise ValueError('unsupported all_reduce alg: ', alg)

  result = []
  for (_, v), g in zip(grad_and_vars, summed_grads):
    result.append([g, v])
  return result


def contains_any(haystack, needles):
  """Tests if any needle is a substring of haystack.

  Args:
    haystack: a string
    needles: list of strings

  Returns:
    True if any element of needles is a substring of haystack,
      False otherwise.
  """
  for n in needles:
    if n in haystack:
      return True
  return False


def sum_gradients_all_reduce(dev_prefixes, tower_grads, num_workers,
                             alg, num_shards, gpu_indices):
  """Apply all-reduce algorithm over specified gradient tensors.

  Args:
    dev_prefixes: list of prefix strings to use to generate PS device names.
    tower_grads: the gradients to reduce.
    num_workers: number of worker processes across entire job.
    alg: the all-reduce algorithm to apply.
    num_shards: alg-specific sharding factor.
    gpu_indices: indices of local GPUs in order usable for ring-reduce.

  Returns:
    list of reduced tensors
  """
  alg_contains_shuffle = contains_any(alg, ['pscpu', 'psgpu'])
  new_tower_grads = []
  is_hierarchical = '/' in alg
  if 'pscpu' in alg:
    aux_devices = [prefix + '/cpu:0' for prefix in dev_prefixes]
  elif 'psgpu' in alg:
    aux_devices = [prefix + '/gpu:%d' % i for i in range(len(gpu_indices))
                   for prefix in dev_prefixes]
  else:
    aux_devices = ['/job:localhost/cpu:0']
  aux_device_groups = group_device_names(
      aux_devices, num_shards if alg_contains_shuffle else 1)
  group_index = 0
  for grad_and_vars in zip(*tower_grads):
    new_tower_grads.append(sum_grad_and_var_all_reduce(
        grad_and_vars, num_workers, alg, gpu_indices,
        aux_devices if is_hierarchical else aux_device_groups[group_index],
        num_shards))
    group_index = (group_index + 1) % len(aux_device_groups)
  return [list(x) for x in zip(*new_tower_grads)]


def aggregate_gradients_using_copy_with_device_selection(
    benchmark_cnn, tower_grads, use_mean):
  """Aggregate gradients, controlling device for the aggregation.

  Args:
    benchmark_cnn: benchmark_cnn class.
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients.
    use_mean: if True, mean is taken, else sum of gradients is taken.

  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers. For each gradient, the variable is chosen from the
     first tower.
  """
  if benchmark_cnn.local_parameter_device_flag == 'gpu':
    avail_devices = benchmark_cnn.raw_devices
  else:
    avail_devices = [benchmark_cnn.param_server_device]
  agg_grads = []
  for i, single_grads in enumerate(zip(*tower_grads)):
    with tf.device(avail_devices[i % len(avail_devices)]):
      agg_grads.append(
          aggregate_single_gradient_using_copy(single_grads, use_mean))
  return agg_grads


def aggregate_gradients_using_copy_with_variable_colocation(
    tower_grads, use_mean):
  """Aggregate gradients, colocating computation with the gradient's variable.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients. All variables
      of the same gradient across towers must be the same (that is,
      tower_grads[x][a][1] == tower_grads[y][a][1] for all indices x, y, and a)
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  agg_grads = []
  for single_grads in zip(*tower_grads):
    # Note that each single_grads looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    var = single_grads[0][1]

    for _, v in single_grads:
      assert v == var

    with tf.device(var.device):
      agg_grads.append(
          aggregate_single_gradient_using_copy(single_grads, use_mean))
  return agg_grads


def aggregate_gradients_using_copy(tower_grads, use_mean):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over towers. The inner list is over individual gradients.
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers. For each gradient, the variable is chosen from the
     first tower.
  """
  return [aggregate_single_gradient_using_copy(single_grads, use_mean)
          for single_grads in zip(*tower_grads)]


def aggregate_single_gradient_using_copy(grad_and_vars, use_mean):
  """Calculate the average gradient for a shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single tower, and the number of pairs
      equals the number of towers.
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
     The pair (average_gradient, variable) where the gradient has been averaged
     across all towers. The variable is chosen from the first tower.
  """
  grads = [g for g, _ in grad_and_vars]
  grad = tf.add_n(grads)

  if use_mean and len(grads) > 1:
    grad = tf.multiply(grad, 1.0/len(grads))

  v = grad_and_vars[0][1]
  return grad, v
