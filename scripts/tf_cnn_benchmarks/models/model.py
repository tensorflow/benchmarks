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
import tensorflow as tf

import convnet_builder


class Model(object):
  """Base model config for DNN benchmarks."""

  def __init__(self, model_name, batch_size, learning_rate, fp16_loss_scale):
    self.model = model_name
    self.batch_size = batch_size
    self.default_batch_size = batch_size
    self.learning_rate = learning_rate
    # TODO(reedwm) Set custom loss scales for each model instead of using the
    # default of 128.
    self.fp16_loss_scale = fp16_loss_scale

  def get_model(self):
    return self.model

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

  def build_network(self, images, phase_train=True, nclass=1001, image_depth=3,
                    data_type=tf.float32, data_format='NCHW',
                    use_tf_layers=True, fp16_vars=False):
    """Builds the forward pass of the model.

    Args:
      images: The images, in NHWC format.
      phase_train: True during training. False during evaluation.
      nclass: Number of classes that the images can belong to.
      image_depth: The channel dimension of `images`. Should be
        `images.shape[3]`.
      data_type: The dtype to run the model in: tf.float32 or tf.float16. The
        variable dtype is controlled by a separate parameter: fp16_vars.
      data_format: What data format to run the model in: NHWC or NCHW.
      use_tf_layers: If True, build the model using tf.layers.
      fp16_vars: If True, the variables will be created in float16.

    Returns:
      logits: The logits of the model.
      aux_logits: The auxiliary logits of the model (see Inception for an
        example), or None if the model does not have auxiliary logits
    """
    raise NotImplementedError('Must be implemented in derived classes')

  def loss_function(self, logits, labels, aux_logits):
    """Loss function for this model."""
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


class CNNModel(Model):
  """Base model configuration for CNN benchmarks."""

  def __init__(self,
               model,
               image_size,
               batch_size,
               learning_rate,
               layer_counts=None,
               fp16_loss_scale=128):
    super(CNNModel, self).__init__(model, batch_size, learning_rate,
                                   fp16_loss_scale)
    self.image_size = image_size
    self.layer_counts = layer_counts

  def get_image_size(self):
    return self.image_size

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

  def build_network(self, images, phase_train=True, nclass=1001, image_depth=3,
                    data_type=tf.float32, data_format='NCHW',
                    use_tf_layers=True, fp16_vars=False):
    """Returns logits and aux_logits from images."""
    if data_format == 'NCHW':
      images = tf.transpose(images, [0, 3, 1, 2])
    var_type = tf.float32
    if data_type == tf.float16 and fp16_vars:
      var_type = tf.float16
    network = convnet_builder.ConvNetBuilder(
        images, image_depth, phase_train, use_tf_layers,
        data_format, data_type, var_type)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      self.add_inference(network)
      # Add the final fully-connected class layer
      logits = (network.affine(nclass, activation='linear')
                if not self.skip_final_affine_layer()
                else network.top_layer)
      aux_logits = None
      if network.aux_top_layer is not None:
        with network.switch_to_aux_top_layer():
          aux_logits = network.affine(
              nclass, activation='linear', stddev=0.001)
    if data_type == tf.float16:
      # TODO(reedwm): Determine if we should do this cast here.
      logits = tf.cast(logits, tf.float32)
      if aux_logits is not None:
        aux_logits = tf.cast(aux_logits, tf.float32)
    return logits, aux_logits
