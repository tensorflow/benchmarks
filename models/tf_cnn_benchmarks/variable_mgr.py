"""Defines VariableMgr and subclasses used to manage variables.
"""

from __future__ import print_function

import operator

import tensorflow as tf

from tensorflow.contrib import nccl
from tensorflow.python.ops import data_flow_ops

PS_SHADOW_VAR_PREFIX = 'ps_var'


# To be used with custom_getter on tf.get_variable.
class OverrideCachingDevice(object):

  def __init__(self, devices):
    self.devices = devices
    self.sizes = [0] * len(self.devices)

  def __call__(self, getter, *args, **kwargs):
    size = tf.TensorShape(kwargs['shape']).num_elements()
    device_index, _ = min(enumerate(
        self.sizes), key=operator.itemgetter(1))
    device_name = self.devices[device_index]
    self.sizes[device_index] += size

    kwargs['caching_device'] = device_name
    var = getter(*args, **kwargs)
    return var


# To be used with custom_getter on tf.get_variable. Ensures the created variable
# is in LOCAL_VARIABLES and not GLOBAL_VARIBLES collection.
#
# TODO(cwhipkey): only trainable variables here?
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

    # TODO(cwhipkey): check that this is compatible with batch-norm.
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
  """VariableMgr that implements the --send_recv mode for local jobs.

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
    return average_gradients_inception(device_grads)

  def get_devices(self):
    raw_devices = self.benchmark_cnn.raw_devices
    if self.benchmark_cnn.parameter_server_flag == 'gpu':
      return [ParamServerDeviceSetter(d, raw_devices) for d in raw_devices]
    else:
      return [tf.train.replica_device_setter(
          worker_device=d, ps_device=self.benchmark_cnn.param_server_device,
          ps_tasks=1) for d in raw_devices]


class StagedVariableGetter(object):
  """A variable getter through staging buffers on devices. Instead of a
  caching device, this getter tracks where the variable is used. And on each
  device, it goes through a staging buffer.
  """

  def __init__(self, device_num, devices, staging_vars_on_devices):
    self.device_num = device_num
    self.devices = devices
    self.staging_vars_on_devices = staging_vars_on_devices

  def __call__(self, getter, name, *args, **kwargs):
    staging_ops = self.staging_vars_on_devices[self.device_num]
    if name in staging_ops:
      put_op, get_op = staging_get_ops[name]
      return get_op
    real_var = getter(name, *args, **kwargs)
    with tf.device(self.devices[self.device_num]):
      shape = kwargs['shape']
      dtype = kwargs['dtype']
      trainable = kwargs['trainable']
      if not trainable:
        # Ignore non-trainable local variables for now.
        return real_var
      staging_area = data_flow_ops.StagingArea([dtype], shapes=[shape])
      put_op = staging_area.put([real_var])
      get_op = staging_area.get()
      staging_ops[name] = (put_op, get_op)
      return get_op


class VariableMgrLocalFetchFromStagedPS(VariableMgrLocalFetchFromPS):
  """VariableMgr that implements fetching a local variable through staging
  buffers.
  """

  def __init__(self, benchmark_cnn):
    super(VariableMgrLocalFetchFromStagedPS, self).__init__(benchmark_cnn)
    self.devices = self.get_devices()
    # A data structure to track where the variables are used on each device.
    # Indexed by device_num and var_name, each entry stores the "put" and "get"
    # ops used for that variable on that device:
    #   staging_vars_on_devices[device_num][var_name] == (put_op, get_op)
    self.staging_vars_on_devices = [dict() for _ in self.devices]

  def supports_staged_vars(self):
    return True

  def create_outer_variable_scope(self, device_num):
    custom_getter = StagedVariableGetter(device_num, self.devices,
                                         self.staging_vars_on_devices)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=custom_getter)


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     nccl all-reduce is used to replicate the combined gradients to
     all towers.
  """

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num)

  def preprocess_device_grads(self, device_grads):
    # Note: passing [] for extra_nccl_ops.  We assume that gradients that are
    # actually needed are used by the optimizer on all devices, so
    # there is no need to track all extra_nccl_ops.
    aggregated_device_grads = average_gradients_all_reduce(
        device_grads, self.benchmark_cnn.devices, extra_nccl_ops=[])
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
    ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      if split_name[0] == 'v0':
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      ops.append(v.assign(copy_from.read_value()))
    return ops

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrDistributedFetchFromPS(VariableMgr):
  """VariableMgr that implements the --send_recv mode for distributed jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    if self.benchmark_cnn.parameter_server_flag == 'gpu':
      custom_getter = OverrideCachingDevice(self.benchmark_cnn.raw_devices)
    else:
      custom_getter = None
    return tf.variable_scope('v', reuse=bool(device_num),
                             custom_getter=custom_getter)

  def preprocess_device_grads(self, device_grads):
    # Returns (gradient_devices, gradient_state)
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    return average_gradients_inception(gradient_state)

  def get_devices(self):
    ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
        len(self.benchmark_cnn.ps_hosts), tf.contrib.training.byte_size_load_fn)
    return [tf.train.replica_device_setter(
        worker_device=d, cluster=self.benchmark_cnn.cluster,
        ps_strategy=ps_strategy)
            for d in self.benchmark_cnn.raw_devices]


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

    # TODO(cwhipkey): make an option to use nccl allreduce here, by making that
    # configuration for that orthogonal to the configuration for variable
    # maintenance.
    avg_grads = average_gradients_inception(device_grads)

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
          # TODO(cwhipkey): allow using nccl broadcast for this.
          for my_d in range(len(self.benchmark_cnn.devices)):
            training_ops.append(
                device_grads[my_d][i][1].assign(updated_value))

  def get_post_init_ops(self):
    # Copy initialized variables for variables on the parameter server
    # to the local copy of the variable.
    #
    # TODO(cwhipkey): verify MomentumOptimizer works correctly with the shadow
    # variables?
    # TODO(cwhipkey): batchnorm variables won't get shadowed to ps server.
    # Does that matter?
    # TODO(cwhipkey): clean up operations that manipulate low-level names to
    # make sure the names and prefixes the script uses are unique to it, and
    # we are not relying on any tensorflow-given names.
    def strip_port(s):
      if s.endswith(':0'):
        return s[:-2]
      return s
    local_vars = tf.local_variables()
    local_var_by_name = dict([(strip_port(v.name), v) for v in local_vars])
    ops = []
    for v in tf.global_variables():
      if v.name.startswith(PS_SHADOW_VAR_PREFIX + '/v0/'):
        prefix = strip_port(
            v.name[len(PS_SHADOW_VAR_PREFIX + '/v0'):])
        for i in range(self.benchmark_cnn.num_gpus):
          name = 'v%s%s' % (i, prefix)
          if name in local_var_by_name:
            copy_to = local_var_by_name[name]
            ops.append(copy_to.assign(v.read_value()))
    return ops

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


def average_grad_and_var_all_reduce(grad_and_vars, devices, extra_nccl_ops):
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

  scaled_grads = [g for _, (g, _) in zip(devices, grad_and_vars)]
  summed_grads = nccl.all_sum(scaled_grads)
  extra_nccl_ops.extend(summed_grads)

  result = []
  for d, (_, v), g in zip(devices, grad_and_vars, summed_grads):
    with tf.device(d):
      result.append((g, v))
  return result


def average_gradients_all_reduce(tower_grads, devices, extra_nccl_ops):
  new_tower_grads = []
  for grad_and_vars in zip(*tower_grads):
    new_tower_grads.append(average_grad_and_var_all_reduce(
        grad_and_vars, devices, extra_nccl_ops))
  return list(zip(*new_tower_grads))


def average_gradients_inception(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
