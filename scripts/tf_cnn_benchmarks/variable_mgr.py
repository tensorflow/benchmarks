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

import operator

import tensorflow as tf

from tensorflow.contrib import nccl
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops

PS_SHADOW_VAR_PREFIX = 'ps_var'


# To be used with custom_getter on tf.get_variable.
class OverrideCachingDevice(object):

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
      device_grads: a list of gradients each of which calculated by a device.

    Returns: a tuple of (apply_gradients_devices, gradient_state), where
      gradients will then be applied to each entry in apply_gradients_devices,
      and gradient is passed to later calls to get_gradients_to_apply and
      append_apply_gradients_ops.
    """
    del device_grads  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_gradients_to_apply(self, device_num, gradient_state):
    """Returns the [(gradient, variable] to apply for device_num.

    Args:
      device_num: indexes ino the apply_gradients_devices returned by an earlier
                  call to preprocess_device_grads.
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

  def retain_tower_updates(self, device_num):
    """Return if only updates for the first GPU tower should be applied."""
    return device_num == 0 and not self.each_tower_has_variables()

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def trainable_variables_on_device(self, device_num, writable=False):
    """Return the set of trainable variables on device.

    Args:
      device_num: the index to the device.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable vairalbes on the specified device.
    """
    del writable
    if self.each_tower_has_variables():
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('v%s/' % device_num)
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
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    return [grad_and_vars[device_num] for grad_and_vars in zip(*device_grads)]

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
      delta_get_op = delta_staging_area.get()
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
      var_to_stage = real_var

    with tf.device(self.devices[self.device_num]):
      staging_area = data_flow_ops.StagingArea([dtype], shapes=[shape])
      put_op = staging_area.put([var_to_stage])
      get_op = staging_area.get()
      staging_ops[name] = (put_op, get_op)
    if trainable:
      # For trainable variables, they are managed separatedly through
      # apply_gradients.
      return get_op
    else:
      # For other shadow variables, the access is decoupled through a wrapper
      # class.
      return StagedModelVariable(real_var, get_op, self.variable_mgr)

  def trainable_variables_on_device(self, device_num, writable):
    """Return the set of trainable variables on the specified device.

    Args:
      device_num: the specified device index.
      writable: whether the returned variables is writable or read-only.

    Returns:
      Return the set of trainable variables on the specified device.
    """
    params_refs = tf.trainable_variables()
    if writable:
      return params_refs
    params = []
    for param in params_refs:
      var_name = param.name.split(':')[0]
      _, var_get_op = self.variable_mgr.staging_vars_on_devices[device_num][
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

  def trainable_variables_on_device(self, device_num, writable=False):
    return self._custom_getter.trainable_variables_on_device(
        device_num, writable=writable)


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either nccl all-reduce or a regular cross-device aggregation is used to
     replicate the combined gradients to all towers.
  """

  def __init__(self, benchmark_cnn, use_nccl):
    super(VariableMgrLocalReplicated, self).__init__(benchmark_cnn)
    self._use_nccl = use_nccl

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num)

  def preprocess_device_grads(self, device_grads):
    if self._use_nccl:
      aggregated_device_grads = sum_gradients_all_reduce(
          device_grads, self.benchmark_cnn.devices)
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
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    return [grad_and_vars[device_num] for grad_and_vars in zip(*device_grads)]

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    return post_init_ops

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

  def trainable_variables_on_device(self, device_num, writable=False):
    return self._custom_getter.trainable_variables_on_device(
        device_num, writable=writable)


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

    # Make shadow variable for each original trainable variable.
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

  def get_post_init_ops(self):
    # Copy initialized variables for variables on the parameter server
    # to the local copy of the variable.
    def strip_port(s):
      if s.endswith(':0'):
        return s[:-2]
      return s
    local_vars = tf.local_variables()
    local_var_by_name = dict([(strip_port(v.name), v) for v in local_vars])
    post_init_ops = []
    for v in tf.global_variables():
      if v.name.startswith(PS_SHADOW_VAR_PREFIX + '/v0/'):
        prefix = strip_port(
            v.name[len(PS_SHADOW_VAR_PREFIX + '/v0'):])
        for i in range(self.benchmark_cnn.num_gpus):
          name = 'v%s%s' % (i, prefix)
          if name in local_var_by_name:
            copy_to = local_var_by_name[name]
            post_init_ops.append(copy_to.assign(v.read_value()))
    return post_init_ops

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


def sum_grad_and_var_all_reduce(grad_and_vars, devices):
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

  scaled_grads = [g for _, (g, _) in zip(devices, grad_and_vars)]
  summed_grads = nccl.all_sum(scaled_grads)

  result = []
  for d, (_, v), g in zip(devices, grad_and_vars, summed_grads):
    with tf.device(d):
      result.append((g, v))
  return result


def sum_gradients_all_reduce(tower_grads, devices):
  new_tower_grads = []
  for grad_and_vars in zip(*tower_grads):
    new_tower_grads.append(sum_grad_and_var_all_reduce(grad_and_vars, devices))
  return list(zip(*new_tower_grads))


def aggregate_gradients_using_copy_with_device_selection(
    benchmark_cnn, tower_grads, use_mean):
  """Aggregate gradients, controlling device for the aggregation.

  Args:
    benchmark_cnn: benchmark_cnn class.
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  if benchmark_cnn.local_parameter_device_flag == 'gpu':
    avail_devices = benchmark_cnn.raw_devices
  else:
    avail_devices = [benchmark_cnn.param_server_device]
  agg_grads = []
  for i, single_grads in enumerate(zip(*tower_grads)):
    with tf.device(avail_devices[i % len(avail_devices)]):
      agg_grads.extend(
          aggregate_gradients_using_copy(zip(single_grads), use_mean))
  return agg_grads


def aggregate_gradients_using_copy_with_variable_colocation(
    tower_grads, use_mean):
  """Aggregate gradients, colocating computation with the gradient's variable.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  agg_grads = []
  for _, single_grads in enumerate(zip(*tower_grads)):
    var = single_grads[0][1]

    for _, v in single_grads:
      assert v == var

    with tf.device(var.device):
      agg_grads.extend(
          aggregate_gradients_using_copy(zip(single_grads), use_mean))
  return agg_grads


def aggregate_gradients_using_copy(tower_grads, use_mean):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    use_mean: if True, mean is taken, else sum of gradients is taken.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  agg_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    grads = [g for g, _ in grad_and_vars]
    grad = tf.add_n(grads)

    if use_mean and len(grads) > 1:
      grad = tf.multiply(grad, 1.0/len(grads))

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    agg_grads.append(grad_and_var)
  return agg_grads
