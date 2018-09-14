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
"""Base model configuration for CNN benchmarks."""

from collections import namedtuple

import tensorflow as tf

import convnet_builder

# BuildNetworkResult encapsulate the result (e.g. logits) of a
# Model.build_network() call.
BuildNetworkResult = namedtuple(
    'BuildNetworkResult',
    [
        'logits',  # logits of the network
        'extra_info',  # Model specific extra information
    ])


class Model(object):
  """Base model config for DNN benchmarks."""

  def __init__(self,
               model_name,
               batch_size,
               learning_rate,
               fp16_loss_scale,
               params=None):
    self.model_name = model_name
    self.batch_size = batch_size
    self.default_batch_size = batch_size
    self.learning_rate = learning_rate
    # TODO(reedwm) Set custom loss scales for each model instead of using the
    # default of 128.
    self.fp16_loss_scale = fp16_loss_scale

    # use_tf_layers specifies whether to build the model using tf.layers.
    # fp16_vars specifies whether to create the variables in float16.
    if params:
      self.use_tf_layers = params.use_tf_layers
      self.fp16_vars = params.fp16_vars
    else:
      self.use_tf_layers = True
      self.fp16_vars = False

  def get_model_name(self):
    return self.model_name

  def get_batch_size(self):
    return self.batch_size

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size

  def get_default_batch_size(self):
    return self.default_batch_size

  def get_fp16_loss_scale(self):
    return self.fp16_loss_scale

  def get_learning_rate(self, global_step, batch_size):
    del global_step
    del batch_size
    return self.learning_rate

  def get_labels_shape(self):
    """Returns the expected shape of labels to this model."""
    return None

  def get_input_shape(self):
    """Returns the expected shape of the input to this model."""
    raise NotImplementedError('Must be implemented in derived classes')

  def get_synthetic_inputs_and_labels(self, input_name, data_type, nclass):
    """Returns the ops to generate synthetic inputs and labels."""
    raise NotImplementedError('Must be implemented in derived classes')

  def build_network(self, inputs, phase_train, nclass, data_type=tf.float32):
    """Builds the forward pass of the model.

    Args:
      inputs: The inputs
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the inputs can belong to.
      data_type: The dtype to run the model in: tf.float32 or tf.float16. The
        variable dtype is controlled by a separate parameter: self.fp16_vars.

    Returns:
      A BuildNetworkResult which contains the logits and model-specific extra
        information.
    """
    raise NotImplementedError('Must be implemented in derived classes')

  def loss_function(self, build_network_result, labels):
    """Returns the op to measure the loss of the model."""
    raise NotImplementedError('Must be implemented in derived classes')

  def accuracy_function(self, logits, labels, data_type):
    """Returns the ops to measure the accuracy of the model."""
    raise NotImplementedError('Must be implemented in derived classes')


class CNNModel(Model):
  """Base model configuration for CNN benchmarks."""

  # TODO(laigd): reduce the number of parameters and read everything from
  # params.
  def __init__(self,
               model,
               image_size,
               batch_size,
               learning_rate,
               layer_counts=None,
               fp16_loss_scale=128,
               params=None):
    super(CNNModel, self).__init__(
        model, batch_size, learning_rate, fp16_loss_scale, params=params)
    self.image_size = image_size
    self.layer_counts = layer_counts
    self.depth = 3
    self.params = params
    self.data_format = params.data_format if params else 'NCHW'

  def get_labels_shape(self):
    return [self.get_batch_size()]

  def get_layer_counts(self):
    return self.layer_counts

  def skip_final_affine_layer(self):
    """Returns if the caller of this class should skip the final affine layer.

    Normally, this class adds a final affine layer to the model after calling
    self.add_inference(), to generate the logits. If a subclass override this
    method to return True, the caller should not add the final affine layer.

    This is useful for tests.
    """
    return False

  def add_backbone_saver(self):
    """Creates a tf.train.Saver as self.backbone_saver for loading backbone.

    A tf.train.Saver must be created and saved in self.backbone_saver before
    calling load_backbone_model, with correct variable name mapping to load
    variables from checkpoint correctly into the current model.
    """
    raise NotImplementedError(self.getName() + ' does not have backbone model.')

  def load_backbone_model(self, sess, backbone_model_path):
    """Loads variable values from a pre-trained backbone model.

    This should be used at the beginning of the training process for transfer
    learning models using checkpoints of base models.

    Args:
      sess: session to train the model.
      backbone_model_path: path to backbone model checkpoint file.
    """
    del sess, backbone_model_path
    raise NotImplementedError(self.getName() + ' does not have backbone model.')

  def add_inference(self, cnn):
    """Adds the core layers of the CNN's forward pass.

    This should build the forward pass layers, except for the initial transpose
    of the images and the final Dense layer producing the logits. The layers
    should be build with the ConvNetBuilder `cnn`, so that when this function
    returns, `cnn.top_layer` and `cnn.top_size` refer to the last layer and the
    number of units of the layer layer, respectively.

    Args:
      cnn: A ConvNetBuilder to build the forward pass layers with.
    """
    del cnn
    raise NotImplementedError('Must be implemented in derived classes')

  def get_input_shape(self):
    # (height, width, depth)
    return [self.image_size, self.image_size, self.depth]

  def get_synthetic_inputs_and_labels(self, input_name, data_type, nclass):
    # Synthetic input should be within [0, 255].
    inputs = tf.truncated_normal(
        [self.batch_size] + self.get_input_shape(),
        dtype=data_type,
        mean=127,
        stddev=60,
        name=self.model_name + '_synthetic_inputs')
    inputs = tf.contrib.framework.local_variable(inputs, name=input_name)
    labels = tf.random_uniform(
        [self.batch_size],
        minval=0,
        maxval=nclass - 1,
        dtype=tf.int32,
        name=self.model_name + '_synthetic_labels')
    return (inputs, labels)

  def build_network(self,
                    inputs,
                    phase_train=True,
                    nclass=1001,
                    data_type=tf.float32):
    """Returns logits from input images.

    Args:
      inputs: The input images
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the images can belong to.
      data_type: The dtype to run the model in: tf.float32 or tf.float16. The
        variable dtype is controlled by a separate parameter: self.fp16_vars.

    Returns:
      A BuildNetworkResult which contains the logits and model-specific extra
        information.
    """
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    var_type = tf.float32
    if data_type == tf.float16 and self.fp16_vars:
      var_type = tf.float16
    network = convnet_builder.ConvNetBuilder(
        inputs, self.depth, phase_train, self.use_tf_layers, self.data_format,
        data_type, var_type)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      self.add_inference(network)
      # Add the final fully-connected class layer
      logits = (
          network.affine(nclass, activation='linear')
          if not self.skip_final_affine_layer() else network.top_layer)
      aux_logits = None
      if network.aux_top_layer is not None:
        with network.switch_to_aux_top_layer():
          aux_logits = network.affine(nclass, activation='linear', stddev=0.001)
    if data_type == tf.float16:
      # TODO(reedwm): Determine if we should do this cast here.
      logits = tf.cast(logits, tf.float32)
      if aux_logits is not None:
        aux_logits = tf.cast(aux_logits, tf.float32)
    return BuildNetworkResult(
        logits=logits, extra_info=None if aux_logits is None else aux_logits)

  def loss_function(self, build_network_result, labels):
    """Returns the op to measure the loss of the model."""
    logits = build_network_result.logits
    # TODO(laigd): consider putting the aux logit logic in the Inception model,
    # which could call super.loss_function twice, once with the normal logits
    # and once with the aux logits.
    aux_logits = build_network_result.extra_info
    with tf.name_scope('xentropy'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    if aux_logits is not None:
      with tf.name_scope('aux_xentropy'):
        aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=aux_logits, labels=labels)
        aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
        loss = tf.add_n([loss, aux_loss])
    return loss

  def accuracy_function(self, logits, labels, data_type):
    """Returns the ops to measure the accuracy of the model."""
    top_1_op = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
    top_5_op = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
    return {'top_1_accuracy': top_1_op, 'top_5_accuracy': top_5_op}
