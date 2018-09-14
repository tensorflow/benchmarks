# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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


"""SSD300 Model Configuration.

References:
  Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
  Cheng-Yang Fu, Alexander C. Berg
  SSD: Single Shot MultiBox Detector
  arXiv:1512.02325

Ported from MLPerf reference implementation:
  https://github.com/mlperf/reference/tree/ssd/single_stage_detector/ssd

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

import ssd_constants
from models import model as model_lib
from models import resnet_model

BACKBONE_MODEL_SCOPE_NAME = 'resnet34_backbone'


class SSD300Model(model_lib.CNNModel):
  """Single Shot Multibox Detection (SSD) model for 300x300 image datasets."""

  def __init__(self, label_num=ssd_constants.NUM_CLASSES, batch_size=32,
               learning_rate=1e-3, backbone='resnet34', params=None):
    super(SSD300Model, self).__init__('ssd300', 300, batch_size, learning_rate)
    # For COCO dataset, 80 categories + 1 background = 81 labels
    # however in dataset there are labels up to 90...
    self.label_num = label_num

    # Currently only support ResNet-34 as backbone model
    if backbone != 'resnet34':
      raise ValueError('Invalid backbone model %s for SSD.' % backbone)

    # Number of channels and default boxes associated with the following layers:
    #   ResNet34 layer, Conv7, Conv8_2, Conv9_2, Conv10_2, Conv11_2
    self.out_chan = [256, 512, 512, 256, 256, 256]

    # Number of default boxes from layers of different scales
    #   38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    self.num_dboxes = [4, 6, 6, 6, 4, 4]

    self.backbone_saver = None

  def skip_final_affine_layer(self):
    return True

  def add_backbone_model(self, cnn):
    # --------------------------------------------------------------------------
    # Resnet-34 backbone model -- modified for SSD
    # --------------------------------------------------------------------------

    # Input 300x300, output 150x150
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
    cnn.mpool(3, 3, 2, 2, mode='SAME')

    resnet34_layers = [3, 4, 6, 3]
    version = 'v1'

    # ResNet-34 block group 1
    # Input 150x150, output 75x75
    for i in range(resnet34_layers[0]):
      # Last argument forces residual_block to use projection shortcut, even
      # though the numbers of input and output channels are equal
      resnet_model.residual_block(cnn, 64, 1, version, i == 0)

    # ResNet-34 block group 2
    # Input 75x75, output 38x38
    for i in range(resnet34_layers[1]):
      stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 128, stride, version, i == 0)

    # ResNet-34 block group 3
    # This block group is modified: first layer uses stride=1 so that the image
    # size does not change in group of layers
    # Input 38x38, output 38x38
    for i in range(resnet34_layers[2]):
      # The following line is intentionally commented out to differentiate from
      # the original ResNet-34 model
      # stride = 2 if i == 0 else 1
      resnet_model.residual_block(cnn, 256, stride, version, i == 0)

    # ResNet-34 block group 4: removed final block group
    # The following 3 lines are intentially commented out to differentiate from
    # the original ResNet-34 model
    # for i in range(resnet34_layers[3]):
    #   stride = 2 if i == 0 else 1
    #   resnet_model.residual_block(cnn, 512, stride, version, i == 0)

  def add_inference(self, cnn):
    # TODO(haoyuzhang): check batch norm params for resnet34 in reference model?
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}

    with tf.variable_scope(BACKBONE_MODEL_SCOPE_NAME):
      self.add_backbone_model(cnn)

    # --------------------------------------------------------------------------
    # SSD additional layers
    # --------------------------------------------------------------------------

    def add_ssd_layer(cnn, depth, k_size, stride, mode):
      return cnn.conv(depth, k_size, k_size, stride, stride,
                      mode=mode, bias=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Activations for feature maps of different layers
    self.activations = [cnn.top_layer]
    # Conv7_1, Conv7_2
    # Input 38x38, output 19x19
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv8_1, Conv8_2
    # Input 19x19, output 10x10
    add_ssd_layer(cnn, 256, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 512, 3, 2, 'same'))

    # Conv9_1, Conv9_2
    # Input 10x10, output 5x5
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 2, 'same'))

    # Conv10_1, Conv10_2
    # Input 5x5, output 3x3
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    # Conv11_1, Conv11_2
    # Input 3x3, output 1x1
    add_ssd_layer(cnn, 128, 1, 1, 'valid')
    self.activations.append(add_ssd_layer(cnn, 256, 3, 1, 'valid'))

    self.loc = []
    self.conf = []

    for nd, ac, oc in zip(self.num_dboxes, self.activations, self.out_chan):
      self.loc.append(tf.reshape(
          cnn.conv(nd * 4, 3, 3, 1, 1, input_layer=ac,
                   num_channels_in=oc, activation=None, bias=None,
                   kernel_initializer=tf.contrib.layers.xavier_initializer()),
          [ac.get_shape()[0], 4, -1]))
      self.conf.append(tf.reshape(
          cnn.conv(nd * self.label_num, 3, 3, 1, 1, input_layer=ac,
                   num_channels_in=oc, activation=None, bias=None,
                   kernel_initializer=tf.contrib.layers.xavier_initializer()),
          [ac.get_shape()[0], self.label_num, -1]))

    # Shape of locs: [batch_size, 4, NUM_SSD_BOXES]
    # Shape of confs: [batch_size, label_num, NUM_SSD_BOXES]
    locs, confs = tf.concat(self.loc, 2), tf.concat(self.conf, 2)

    # Pack location and confidence outputs into a single output layer
    # Shape of logits: [batch_size, 4+label_num, NUM_SSD_BOXES]
    logits = tf.concat([locs, confs], 1)

    cnn.top_layer = logits
    cnn.top_size = 4 + self.label_num

    return cnn.top_layer

  def get_learning_rate(self, global_step, batch_size):
    boundaries = [160000, 200000]
    learning_rates = [1e-3, 1e-4, 1e-5]
    return tf.train.piecewise_constant(global_step, boundaries, learning_rates)

  def _collect_backbone_vars(self):
    backbone_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='.*'+ BACKBONE_MODEL_SCOPE_NAME)
    var_list = {}

    # Assume variables in the checkpoint are following the naming convention of
    # a model checkpoint trained with TF official model
    # TODO(haoyuzhang): the following variable name parsing is hacky and easy
    # to break if there is change in naming convention of either benchmarks or
    # official models.
    for v in backbone_vars:
      # conv2d variable example (model <-- checkpoint):
      #   v/cg/conv24/conv2d/kernel:0 <-- conv2d_24/kernel
      if 'conv2d' in v.name:
        re_match = re.search(r'conv(\d+)/conv2d/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'conv2d', layer_id, param_name)
          var_list[vname_in_ckpt] = v

      # batchnorm varariable example:
      #   v/cg/conv24/batchnorm25/gamma:0 <-- batch_normalization_25/gamma
      elif 'batchnorm' in v.name:
        re_match = re.search(r'batchnorm(\d+)/(.+):', v.name)
        if re_match:
          layer_id = int(re_match.group(1))
          param_name = re_match.group(2)
          vname_in_ckpt = self._var_name_in_official_model_ckpt(
              'batch_normalization', layer_id, param_name)
          var_list[vname_in_ckpt] = v

    return var_list

  def _var_name_in_official_model_ckpt(self, layer_name, layer_id, param_name):
    """Return variable names according to convention in TF official models."""
    vname_in_ckpt = layer_name
    if layer_id > 0:
      vname_in_ckpt += '_' + str(layer_id)
    vname_in_ckpt += '/' + param_name
    return vname_in_ckpt

  def loss_function(self, build_network_result, labels):
    try:
      import ssd_dataloader  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs by '
                        'following https://github.com/tensorflow/models/blob/'
                        'master/research/object_detection/g3doc/installation.md'
                        '#protobuf-compilation')

    logits = build_network_result.logits
    # Unpack model output back to locations and confidence scores of predictions
    # Shape of pred_loc: [batch_size, 4, NUM_SSD_BOXES]
    # Shape of pred_label: [batch_size, label_num, NUM_SSD_BOXES]
    pred_loc, pred_label = tf.split(logits, [4, self.label_num], 1)

    # Unpack ground truth labels to number of boxes, locations, and classes
    # initial shape: [batch_size, NUM_SSD_BOXES, 5]
    # Shape of labels: [batch_size, NUM_SSD_BOXES, 5]
    # Shape of num_gt: [batch_size, 1, 5] -- 5 identical copies
    labels, num_gt = tf.split(labels, [ssd_constants.NUM_SSD_BOXES, 1], 1)

    # Shape of num_gt: [batch_size]
    num_gt = tf.squeeze(tf.cast(num_gt[:, :, 0], tf.int32))

    # Shape of gt_loc: [batch_size, NUM_SSD_BOXES, 4]
    # Shape of gt_label: [batch_size, NUM_SSD_BOXES, 1]
    gt_loc, gt_label = tf.split(labels, [4, 1], 2)
    gt_label = tf.cast(gt_label, tf.int32)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        gt_label, tf.transpose(pred_label, [0, 2, 1]),
        reduction=tf.losses.Reduction.NONE)

    default_boxes = tf.tile(tf.convert_to_tensor(
        ssd_dataloader.DefaultBoxes()('xywh'))[tf.newaxis, :, :],
                            [gt_loc.get_shape()[0], 1, 1])

    # To performance people: MLPerf uses this transposed convention.
    # I (taylorrobie) have matched it to make it easier to compare to the
    # reference. If this hurts performance, feel free to adjust accordingly.
    gt_label = tf.squeeze(gt_label)
    # pred_loc, pred_label, gt_loc, default_boxes = [
    #     tf.transpose(i, (0, 2, 1)) for i in
    #     [pred_loc, pred_label, gt_loc, default_boxes]
    # ]

    # Shape of gt_loc: [batch_size, 4, NUM_SSD_BOXES]
    gt_loc = tf.transpose(gt_loc, [0, 2, 1])

    # Shape of default_boxes: [batch_size, 4, NUM_SSD_BOXES]
    default_boxes = tf.transpose(default_boxes, [0, 2, 1])

    mask = tf.greater(gt_label, 0)
    float_mask = tf.cast(mask, tf.float32)

    gt_location_vectors = tf.concat([
        (ssd_constants.SCALE_XY * (gt_loc[:, :2, :] - default_boxes[:, :2, :]) /
         default_boxes[:, 2:, :]),

        # The gt_loc height and width have already had the log taken.
        # See FasterRcnnBoxCoder for more details.
        (ssd_constants.SCALE_HW * (gt_loc[:, 2:, :] -
                                   tf.log(default_boxes[:, 2:, :])))
    ], axis=1)

    smooth_l1 = tf.reduce_sum(tf.losses.huber_loss(
        gt_location_vectors, pred_loc,
        reduction=tf.losses.Reduction.NONE
    ), axis=1)
    smooth_l1 = tf.multiply(smooth_l1, float_mask)
    box_loss = tf.reduce_sum(smooth_l1, axis=1)

    # Hard example mining
    neg_masked_cross_entropy = cross_entropy * (1 - float_mask)

    relative_position = tf.contrib.framework.argsort(
        tf.contrib.framework.argsort(
            neg_masked_cross_entropy, direction='DESCENDING'))
    num_neg_boxes = num_gt * ssd_constants.NEGS_PER_POSITIVE
    top_k_neg_mask = tf.cast(tf.less(
        relative_position,
        tf.tile(num_neg_boxes[:, tf.newaxis], (1, ssd_constants.NUM_SSD_BOXES))
    ), tf.float32)

    class_loss = tf.reduce_sum(
        tf.multiply(cross_entropy, float_mask + top_k_neg_mask), axis=1)

    class_loss = tf.reduce_mean(class_loss / tf.cast(num_gt, tf.float32))
    box_loss = tf.reduce_mean(box_loss / tf.cast(num_gt, tf.float32))

    return class_loss + box_loss

  def add_backbone_saver(self):
    # Create saver with mapping from variable names in checkpoint of backbone
    # model to variables in SSD model
    if not self.backbone_saver:
      backbone_var_list = self._collect_backbone_vars()
      self.backbone_saver = tf.train.Saver(backbone_var_list)

  def load_backbone_model(self, sess, backbone_model_path):
    self.backbone_saver.restore(sess, backbone_model_path)
    return

  def get_labels_shape(self):
    """See ssd_dataloader.py for shape details."""
    return [self.get_batch_size(), ssd_constants.NUM_SSD_BOXES+1, 5]
