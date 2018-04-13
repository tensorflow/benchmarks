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

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
import allreduce
import variable_mgr_util


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCNN to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """

  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []

    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None

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

  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    """Adds training ops for grads to 'training_ops'.



    Args:
      gradient_state: from previous call to apply_gradients_devices.
      opt: the underlying optimizer
      grads: [(grad, var)] to apply
      training_ops: list to which to add ops
      loss_scale_params: parameters for loss scaling.
    """
    del gradient_state  # unused by this implementation

    def get_apply_gradients_ops_func():
      """Returns the apply_gradients op."""
      return [opt.apply_gradients(grads)]

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    """Return the set of trainable variables on device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable variables on the specified device.
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
    tower_grad = device_grads[device_num]

    if self.benchmark_cnn.enable_auto_loss_scale and device_num == 0:
      # Since we don't aggregate variables in --independent mode, we cannot tell
      # if there are NaNs on all GPUs. So we arbitrarily choose to only check
      # NaNs on the first GPU.
      has_inf_nan_list = []
      for grad, _ in tower_grad:
        has_inf_nan_list.append(tf.reduce_all(tf.is_finite(grad)))
      self.grad_has_inf_nan = tf.logical_not(tf.reduce_all(has_inf_nan_list))

    return tower_grad

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
    agg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.
        aggregate_gradients_using_copy_with_variable_colocation(
            device_grads,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    return agg_grads

  def get_devices(self):
    raw_devices = self.benchmark_cnn.raw_devices
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      return [
          variable_mgr_util.ParamServerDeviceSetter(d, raw_devices)
          for d in raw_devices
      ]
    else:
      return [
          tf.train.replica_device_setter(
              worker_device=d,
              ps_device=self.benchmark_cnn.param_server_device,
              ps_tasks=1) for d in raw_devices
      ]


class VariableMgrLocalFetchFromStagedPS(VariableMgrLocalFetchFromPS):
  """Implements fetching a local variable through staging buffers.
  """

  def __init__(self, benchmark_cnn):
    super(VariableMgrLocalFetchFromStagedPS, self).__init__(benchmark_cnn)
    # A data structure to track where the variables are used on each device.
    # Indexed by device_num and var_name, each entry stores the "put" and "get"
    # ops used for that variable on that device:
    #   staging_vars_on_devices[device_num][var_name] == (put_op, get_op)
    self.staging_vars_on_devices = [
        dict() for _ in self.benchmark_cnn.raw_devices
    ]

  def supports_staged_vars(self):
    return True

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = variable_mgr_util.StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices, None, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=self._custom_getter)

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    return self._custom_getter.trainable_variables_on_device(
        rel_device_num, abs_device_num, writable=writable)


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either a local all-reduce algorithm is applied or a regular
     cross-device aggregation is used to replicate the combined
     gradients to all towers.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec, agg_small_grads_max_bytes,
               agg_small_grads_max_group):
    super(VariableMgrLocalReplicated, self).__init__(benchmark_cnn)
    if all_reduce_spec:
      spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
      if len(spec) != 1:
        raise ValueError(
            'replicated mode does not support hybrid all-reduce strategies')
      self._all_reduce_spec = spec[0]
    else:
      self._all_reduce_spec = None
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    self._warmup_ops = []
    self._gradient_put_ops = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num)

  def _aggregate_grads(self, device_grads):
    """Aggregate gradients across GPUs.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns:
      List of lists of (gradient, variable) tuples, in the same form
      as `device_grads`. Each gradient has been summed over the towers.
    """
    if self._all_reduce_spec:
      # TODO(reedwm): Merge allreduce.sum_gradients_all_reduce with the other
      # gradient aggregation code, since gradient aggregation is doing an all
      # reduce. Currently, we do gradient repacking in two different places.
      aggregated_device_grads = allreduce.sum_gradients_all_reduce(
          ['/job:localhost'],
          device_grads,
          1,
          self._all_reduce_spec.alg,
          self._all_reduce_spec.shards,
          self.benchmark_cnn.gpu_indices,
          agg_small_grads_max_bytes=self._agg_small_grads_max_bytes,
          agg_small_grads_max_group=self._agg_small_grads_max_group)
    elif self.benchmark_cnn.params.hierarchical_copy:
      aggregated_device_grads, self.grad_has_inf_nan = (
          variable_mgr_util.aggregate_gradients_using_hierarchical_copy(
              self.benchmark_cnn,
              device_grads,
              use_mean=False,
              check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    else:
      agg_grads, self.grad_has_inf_nan = (
          variable_mgr_util.
          aggregate_gradients_using_copy_with_device_selection(
              self.benchmark_cnn,
              device_grads,
              use_mean=False,
              check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
      aggregated_device_grads = []
      for arr in device_grads:
        aggregated_device_grads.append(
            [(g, v) for (_, v), (g, _) in zip(arr, agg_grads)])
    return aggregated_device_grads

  def preprocess_device_grads(self, device_grads):
    pack_grads = (self.benchmark_cnn.params.gradient_repacking != 0)
    compact_grads = (self.benchmark_cnn.params.use_fp16 and
                     self.benchmark_cnn.params.compact_gradient_transfer)
    defer_grads = (self.benchmark_cnn.params.variable_consistency == 'relaxed')

    # Before aggregating gradients, we do several preprocessing functions that
    # can speed up the aggregation. We undo these functions after aggregating
    # the gradients.
    # TODO(reedwm): Encapsulate the preprocessing functions and undo functions
    # into their own classes.
    if pack_grads:
      device_grads_before_concat = device_grads
      device_grads, grad_states = self._concat_grads(device_grads)
    # If enabled, we compact and defer gradients in between concatenating them
    # and splitting them, because it is faster to do operations on a single
    # concatenated tensor than on multiple smaller tensors.
    if compact_grads:
      device_grads_before_compact = device_grads
      device_grads = self._compact_grads(device_grads)
    if defer_grads:
      device_grads = self._defer_grads(device_grads)
    if pack_grads:
      device_grads_before_split = device_grads
      device_grads = self._split_grads(device_grads)

    device_grads = self._aggregate_grads(device_grads)

    # Undo the preprocessing operations in opposite order as we applied them.
    if pack_grads:
      device_grads = self._undo_split_grads(device_grads,
                                            device_grads_before_split)
    if compact_grads:
      device_grads = self._undo_compact_grads(device_grads,
                                              device_grads_before_compact)
    # Note: There is no undo operation for defer_grads. But we do need to call
    # self._add_put_op_control_deps at the end if we deferred the gradients.
    if pack_grads:
      device_grads = self._undo_concat_grads(device_grads,
                                             device_grads_before_concat,
                                             grad_states)

    if defer_grads:
      device_grads = self._add_put_op_control_deps(device_grads)
    return self.benchmark_cnn.devices, device_grads

  def _defer_gradient(self, grad, tower_num):
    """Defers the retrieval of a gradient.

    The gradient is put into a StagingArea, and the return value is the
    retrieval of the gradient from the StagingArea. The effect is that the
    gradient returned from this function is the gradient computed from the
    previous step.

    The put op is put in self._gradient_put_ops[tower_num], which must be run
    every step. self._gradient_put_ops must be set to a list of lists before
    this function is run. A warmup op to fill the StagingArea with a zero
    gradient is added to self._warmup_ops, which must be run before the first
    step.

    Args:
      grad: The gradient tensor to defer for one step.
      tower_num: The tower that computed the gradient.

    Returns:
      The gradient, deferred for one step.
    """
    gradient_stage = data_flow_ops.StagingArea([grad.dtype], [grad.shape])

    # Push the gradient into the staging area.
    gradient_put_op = gradient_stage.put([grad])
    self._gradient_put_ops[tower_num].append(gradient_put_op)

    # Push an empty gradient into the staging area.
    warmup_op = gradient_stage.put([tf.zeros(grad.shape, dtype=grad.dtype)])
    self._warmup_ops.append(warmup_op)

    # Fetch the next gradient to ues.
    (grad,) = gradient_stage.get()
    return grad

  def _defer_grads(self, device_grads):
    """Defers each gradient in `device_grads`. See `self._defer_gradient`.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.
    Returns:
      `device_grads`, except each gradient has been deferred with
      `self._defer_gradient`.
    """
    self._gradient_put_ops = [[] for _ in device_grads]
    deferred_device_grads = []
    for i, tower_grad_vars in enumerate(device_grads):
      deferred_tower_grad_vars = []
      for g, v in tower_grad_vars:
        with tf.colocate_with(g):
          deferred_tower_grad_vars.append((self._defer_gradient(g, i), v))
      deferred_device_grads.append(deferred_tower_grad_vars)
    return deferred_device_grads

  def _add_put_op_control_deps(self, device_grads):
    """Add control dependencies from self._gradient_put_ops to device_grads.

    This should only be called when deferred gradients are being used.
    Otherwise, there are no put ops.

    The control dependencies are added so that we don't have a race condition
    with the update operation that follows. Also, it causes the put ops to run
    when the gradients are run. Otherwise, the caller has to explicitly run the
    put ops.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.
    Returns:
      `device_grads`, except the gradients are new ops producing the same values
      as before, but have control dependencies on the put ops.

    """
    device_grads_with_deps = []
    for tower_grad_vars, tower_grad_put_ops in zip(device_grads,
                                                   self._gradient_put_ops):
      if self.benchmark_cnn.params.gradient_repacking == 0:
        assert len(tower_grad_put_ops) == len(tower_grad_vars)
        tower_grad_vars = [
            (control_flow_ops.with_dependencies([tower_grad_put_ops[i]], g), v)
            for i, (g, v) in enumerate(tower_grad_vars)
        ]
      else:
        assert len(tower_grad_put_ops) == 1
        tower_grad_vars = [
            (control_flow_ops.with_dependencies(tower_grad_put_ops, g), v)
            for g, v in tower_grad_vars
        ]
      device_grads_with_deps.append(tower_grad_vars)
    return device_grads_with_deps

  def _compact_grads(self, device_grads):
    """Compacts gradients in `device_grads` by casting them to fp16.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.
    Returns:
      `device_grads`, except the gradients have been casted to fp16.
    """
    casted_device_grads = []
    for tower_grad_vars in device_grads:
      casted_tower_grad_vars = []
      for g, v in tower_grad_vars:
        with tf.colocate_with(g):
          casted_tower_grad_vars.append((tf.cast(g, tf.float16), v))
      casted_device_grads.append(casted_tower_grad_vars)
    return casted_device_grads

  def _undo_compact_grads(self, device_grads, orig_device_grads):
    """Undoes the effects of `self._compact_grads`.

    Args:
      device_grads: List of lists of (gradient, variable) tuples, in the same
        form as the return value of `self._compact_grads`.
      orig_device_grads: The original `device_grads` passed to
        `self._compact_grads`.
    Returns:
      `device_grads`, except the gradients have been casted to their original
        dtype.
    """
    new_device_grads = []
    for tower_grad_vars, orig_tower_grad_vars in zip(device_grads,
                                                     orig_device_grads):
      new_tower_grad_vars = []
      for (g, v), (og, _) in zip(tower_grad_vars, orig_tower_grad_vars):
        with tf.colocate_with(og):
          new_tower_grad_vars.append((tf.cast(g, og.dtype), v))
      new_device_grads.append(new_tower_grad_vars)
    return new_device_grads

  def _concat_grads(self, device_grads):
    """Concatenates the gradients in device_grads.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns:
      concatenated_device_grads: List of lists of (gradient, variable) tuples.
        return_value[t] is a list with a single element,
        (concatenated_gradient, None). The return value can be passed to other
        functions taking `device_grads` as an argument.
      grad_states: A list of ConcatGradientStates to be passed to
        `self._undo_concat_grads`
    """
    concatenated_device_grads = []
    grad_states = []
    for tower_grads_and_vars in device_grads:
      grads = [g for g, _ in tower_grads_and_vars]
      with tf.colocate_with(grads[0]):
        concat_grad, grad_state = variable_mgr_util.concat_grads(grads)

      # TODO(zhengxq): It is hacky to have to use fake variables.
      # We should remove the need for variables in
      # aggregate_gradients_using*.
      concatenated_device_grads.append([(concat_grad, None)])
      grad_states.append(grad_state)
    return concatenated_device_grads, grad_states

  def _split_grads(self, concatenated_device_grads):
    """Splits the concatenated device grads.

    Args:
      concatenated_device_grads: Concatenated device grads, in the same form as
        returned by `self._concat_grads`.

    Returns:
      List of lists of (gradient, variable) tuples.
      return_value[t][g] = (gradient, variable), where t is the index of the
      tower and g is the index of the split tensor. The return value can be
      passed to other functions taking `device_grads` as an argument.
    """
    device_grad_packs = []
    for [(concat_grad, _)] in concatenated_device_grads:
      with tf.colocate_with(concat_grad):
        grad_packs = variable_mgr_util.split_grads(
            concat_grad, self.benchmark_cnn.params.gradient_repacking)
        device_grad_packs.append(list(
            zip(grad_packs, [None] * len(grad_packs))))
    return device_grad_packs

  def _undo_split_grads(self, device_grad_packs,
                        orig_concatenated_device_grads):
    """Undoes the effects of `self._split_grads`.

    Arguments:
      device_grad_packs: Aggregated gradients, in the same form as returned by
        `split_grads`.
      orig_concatenated_device_grads: The original `concatenated_device_grads`
        passed to `self._split_grads`.
    Returns:
      Concatenated gradients in the same form as the parameter to
      `self._split_grads`.
    """
    concatenated_device_grads = []
    for tower_grads_and_vars, [(orig_concat_grad, _)] in zip(
        device_grad_packs, orig_concatenated_device_grads):
      grads = [g for g, _ in tower_grads_and_vars]
      with tf.colocate_with(orig_concat_grad):
        concat_grad = variable_mgr_util.undo_split_grads(grads)

      concatenated_device_grads.append([(concat_grad, None)])
    return concatenated_device_grads

  def _undo_concat_grads(self, concatenated_device_grads, orig_device_grads,
                         grad_states):
    """Undoes the effects of `self._concat_grads`.

    Args:
      concatenated_device_grads: Concatenated device grads in the same form as
        returned by `self._undo_split_grads`.
      orig_device_grads: The original `device_grads` passed to
        `self._concat_grads`.
      grad_states:
        The gradient states returned by `self._concat_grads`.
    Returns:
      Unconcatenated gradients, in the same form as the parameter to
      `self._concat_grads`.
    """
    device_grads = []
    for [(concat_grad, _)], orig_tower_grads, grad_state in zip(
        concatenated_device_grads, orig_device_grads, grad_states):
      with tf.colocate_with(concat_grad):
        grads_with_shapes = variable_mgr_util.undo_concat_grads(concat_grad,
                                                                grad_state)
        # Form the list with the original list of variables.
        tower_grad_vars = [
            (g, v) for g, (_, v) in zip(grads_with_shapes, orig_tower_grads)
        ]
        device_grads.append(tower_grad_vars)
    return device_grads

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
    post_init_ops += self._warmup_ops
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

  def __init__(self, benchmark_cnn, all_reduce_spec, job_name, num_workers,
               agg_small_grads_max_bytes, agg_small_grads_max_group):
    super(VariableMgrDistributedAllReduce, self).__init__(benchmark_cnn)
    if not all_reduce_spec:
      raise ValueError(
          'distributed_all_reduce requires a non-empty all_reduce_spec')
    self._all_reduce_spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
    self._all_reduce_device_prefixes = (
        allreduce.build_all_reduce_device_prefixes(job_name, num_workers))
    self._num_workers = num_workers
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
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
        (this_grads, remaining_grads) = allreduce.split_grads_by_size(
            spec_tuple.limit, remaining_grads)
      if this_grads:
        range_agg_grads = allreduce.sum_gradients_all_reduce(
            self._all_reduce_device_prefixes,
            this_grads,
            self._num_workers,
            spec_tuple.alg,
            spec_tuple.shards,
            self.benchmark_cnn.gpu_indices,
            agg_small_grads_max_bytes=self._agg_small_grads_max_bytes,
            agg_small_grads_max_group=self._agg_small_grads_max_group)
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
    custom_getter = variable_mgr_util.OverrideCachingDevice(
        caching_devices, self.benchmark_cnn.cpu_device, 1024 * 64)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=custom_getter)

  def preprocess_device_grads(self, device_grads):
    # Returns (gradient_devices, gradient_state)
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    agg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.aggregate_gradients_using_copy(
            gradient_state,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    return agg_grads

  def get_devices(self):
    ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
        self.benchmark_cnn.num_ps, tf.contrib.training.byte_size_load_fn)
    return [
        tf.train.replica_device_setter(
            worker_device=d,
            cluster=self.benchmark_cnn.cluster_manager.get_cluster_spec(),
            ps_strategy=ps_strategy) for d in self.benchmark_cnn.raw_devices
    ]


class VariableMgrDistributedFetchFromStagedPS(
    VariableMgrDistributedFetchFromPS):
  """Extends VariableMgrDistributedFetchFromPS for --staged_vars."""

  def __init__(self, benchmark_cnn):
    super(VariableMgrDistributedFetchFromStagedPS, self).__init__(benchmark_cnn)
    self.staging_vars_on_devices = [
        dict() for _ in self.benchmark_cnn.raw_devices
    ]
    self.staged_vars_on_cpu = {}

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = variable_mgr_util.StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices,
        self.benchmark_cnn.cpu_device, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num), custom_getter=self._custom_getter)

  def supports_staged_vars(self):
    return True

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
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
        custom_getter=variable_mgr_util.OverrideToLocalVariableIfNotPsVar())

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    avg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.aggregate_gradients_using_copy_with_device_selection(
            self.benchmark_cnn,
            device_grads,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))

    # Make shadow variable on a parameter server for each original trainable
    # variable.
    for i, (g, v) in enumerate(avg_grads):
      my_name = variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/' + v.name
      if my_name.endswith(':0'):
        my_name = my_name[:-2]
      new_v = tf.get_variable(
          my_name,
          dtype=v.dtype.base_dtype,
          initializer=v.initial_value,
          trainable=True)
      avg_grads[i] = (g, new_v)
    return avg_grads

  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    def get_apply_gradients_ops_func():
      """Returns a list of ops for updating gradients."""
      apply_gradients_ops = []
      # For each variable, apply the combined gradients for this server on
      # the parameter server, and then wait for all other servers to do this.
      for i, (g, v) in enumerate(grads):
        apply_gradient_op = opt.apply_gradients([(g, v)])
        barrier = self.benchmark_cnn.add_sync_queues_and_barrier(
            'replicate_variable_%s' % i, [apply_gradient_op])
        with tf.control_dependencies([barrier]):
          with tf.device(self.benchmark_cnn.cpu_device):
            updated_value = v.read_value()
            for my_d in range(len(self.benchmark_cnn.devices)):
              apply_gradients_ops.append(
                  device_grads[my_d][i][1].assign(updated_value))
      return apply_gradients_ops

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

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
      if v.name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0/'):
        prefix = self._strip_port(
            v.name[len(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0'):])
        for i in range(self.benchmark_cnn.num_gpus):
          name = 'v%s%s' % (i, prefix)
          if name in local_var_by_name:
            copy_to = local_var_by_name[name]
            post_init_ops.append(copy_to.assign(v.read_value()))
    return post_init_ops

  def _remove_shadow_var_prefix_if_present(self, var_name):
    if var_name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/'):
      return var_name[len(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/'):]
    else:
      return var_name

  def var_dict_name(self, v):
    return self._strip_port(self._remove_shadow_var_prefix_if_present(v.name))

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    params = {}
    for v in tf.global_variables():
      assert (v.name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0/')
              or v.name in ('global_step:0', 'loss_scale:0',
                            'loss_scale_normal_steps:0')), (
                                'Invalid global variable: %s' % v)
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
