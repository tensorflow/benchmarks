# Copyright 2018 Google. All Rights Reserved.
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
"""Data loader and processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math

import numpy as np
import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.core import box_list
from object_detection.core import preprocessor
from object_detection.core import region_similarity_calculator
from object_detection.core import target_assigner
from object_detection.data_decoders import tf_example_decoder
from object_detection.matchers import argmax_matcher
import ssd_constants


class DefaultBoxes(object):
  """Default bounding boxes for 300x300 5 layer SSD"""

  def __init__(self):
    fk = ssd_constants.IMAGE_SIZE / np.array(ssd_constants.STEPS)

    self.default_boxes = []
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
      sk1 = ssd_constants.SCALES[idx] / ssd_constants.IMAGE_SIZE
      sk2 = ssd_constants.SCALES[idx+1] / ssd_constants.IMAGE_SIZE
      sk3 = math.sqrt(sk1*sk2)
      all_sizes = [(sk1, sk1), (sk3, sk3)]

      for alpha in ssd_constants.ASPECT_RATIOS[idx]:
        w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
        all_sizes.append((w, h))
        all_sizes.append((h, w))

      assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]

      for w, h in all_sizes:
        for i, j in it.product(range(feature_size), repeat=2):
          cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
          box = tuple(np.clip(k, 0, 1) for k in (cx, cy, w, h))
          self.default_boxes.append(box)

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES

    def to_ltrb(cx, cy, w, h):
      return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb': return self.default_boxes_ltrb
    if order == 'xywh': return self.default_boxes


def calc_iou_tensor(box1, box2):
  """ Calculation of IoU based on two boxes tensor,
      Reference to https://github.com/kuangliu/pytorch-ssd
      input:
          box1 (N, 4)
          box2 (M, 4)
      output:
          IoU (N, M)
  """
  N = tf.shape(box1)[0]
  M = tf.shape(box2)[0]

  be1 = tf.tile(tf.expand_dims(box1, axis=1), (1, M, 1))
  be2 = tf.tile(tf.expand_dims(box2, axis=0), (N, 1, 1))

  # Left Top & Right Bottom
  lt = tf.maximum(be1[:, :, :2], be2[:, :, :2])

  rb = tf.minimum(be1[:, :, 2:], be2[:, :, 2:])

  delta = tf.maximum(rb - lt, 0)

  intersect = delta[:, :, 0] * delta[:, :, 1]

  delta1 = be1[:, :, 2:] - be1[:, :, :2]
  area1 = delta1[:, :, 0] * delta1[:, :, 1]
  delta2 = be2[:, :, 2:] - be2[:, :, :2]
  area2 = delta2[:, :, 0] * delta2[:, :, 1]

  iou = intersect/(area1 + area2 - intersect)
  return iou


def ssd_crop(image, boxes, classes):
  """IoU biassed random crop.

  Reference: https://github.com/chauhan-utk/ssd.DomainAdaptation
  """

  num_boxes = tf.shape(boxes)[0]

  def no_crop_check():
    return (tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
            < ssd_constants.P_NO_CROP_PER_PASS)

  def no_crop_proposal():
    return (
        tf.ones((), tf.bool),
        tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32),
        tf.ones((num_boxes,), tf.bool),
    )

  def crop_proposal():
    rand_vec = lambda minval, maxval: tf.random_uniform(
        shape=(ssd_constants.NUM_CROP_PASSES, 1), minval=minval, maxval=maxval,
        dtype=tf.float32)

    width, height = rand_vec(0.3, 1), rand_vec(0.3, 1)
    left, top = rand_vec(0, 1-width), rand_vec(0, 1-height)

    right = left + width
    bottom = top + height

    ltrb = tf.concat([left, top, right, bottom], axis=1)

    min_iou = tf.random_shuffle(ssd_constants.CROP_MIN_IOU_CHOICES)[0]
    ious = calc_iou_tensor(ltrb, boxes)

    # discard any bboxes whose center not in the cropped image
    xc, yc = [tf.tile(0.5 * (boxes[:, i + 0] + boxes[:, i + 2])[tf.newaxis, :],
                      (ssd_constants.NUM_CROP_PASSES, 1)) for i in range(2)]

    masks = tf.reduce_all(tf.stack([
        tf.greater(xc, tf.tile(left, (1, num_boxes))),
        tf.less(xc, tf.tile(right, (1, num_boxes))),
        tf.greater(yc, tf.tile(top, (1, num_boxes))),
        tf.less(yc, tf.tile(bottom, (1, num_boxes))),
    ], axis=2), axis=2)

    # Checks of whether a crop is valid.
    valid_aspect = tf.logical_and(tf.less(height/width, 2),
                                  tf.less(height/width, 2))
    valid_ious = tf.reduce_all(tf.greater(ious, min_iou), axis=1, keepdims=True)
    valid_masks = tf.reduce_any(masks, axis=1, keepdims=True)

    valid_all = tf.cast(tf.reduce_all(tf.concat(
        [valid_aspect, valid_ious, valid_masks], axis=1), axis=1), tf.int32)

    # One indexed, as zero is needed for the case of no matches.
    index = tf.range(1, 1 + ssd_constants.NUM_CROP_PASSES, dtype=tf.int32)

    # Either one-hot, or zeros if there is no valid crop.
    selection = tf.equal(tf.reduce_max(index * valid_all), index)

    use_crop = tf.reduce_any(selection)
    output_ltrb = tf.reduce_sum(tf.multiply(ltrb, tf.tile(tf.cast(
        selection, tf.float32)[:, tf.newaxis], (1, 4))), axis=0)
    output_masks = tf.reduce_any(tf.logical_and(masks, tf.tile(
        selection[:, tf.newaxis], (1, num_boxes))), axis=0)

    return use_crop, output_ltrb, output_masks

  def proposal(*args):
    return tf.cond(
        pred=no_crop_check(),
        true_fn=no_crop_proposal,
        false_fn=crop_proposal,
    )

  _, crop_bounds, box_masks = tf.while_loop(
      cond=lambda x, *_: tf.logical_not(x),
      body=proposal,
      loop_vars=[tf.zeros((), tf.bool), tf.zeros((4,), tf.float32), tf.zeros((num_boxes,), tf.bool)],
  )

  filtered_boxes = tf.boolean_mask(boxes, box_masks, axis=0)
  num_filtered_boxes = tf.shape(filtered_boxes)[0]

  # Clip boxes to the cropped region.
  filtered_boxes = tf.stack([
      tf.maximum(filtered_boxes[:, 0], crop_bounds[0]),
      tf.maximum(filtered_boxes[:, 1], crop_bounds[1]),
      tf.minimum(filtered_boxes[:, 2], crop_bounds[2]),
      tf.minimum(filtered_boxes[:, 3], crop_bounds[3]),
  ], axis=1)

  left = crop_bounds[0]
  top = crop_bounds[1]
  width = crop_bounds[2] - left
  height = crop_bounds[3] - top

  cropped_boxes = tf.stack([
      (filtered_boxes[:, 0] - left) / width,
      (filtered_boxes[:, 1] - top) / height,
      (filtered_boxes[:, 2] - left) / width,
      (filtered_boxes[:, 3] - top) / height,
  ], axis=1)

  cropped_image = tf.image.crop_and_resize(
      image=image[tf.newaxis, :, :, :],
      boxes=crop_bounds[tf.newaxis, :],
      box_ind=tf.zeros((1,), tf.int32),
      crop_size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE),
  )[0, :, :, :]

  cropped_classes = tf.boolean_mask(classes, box_masks, axis=0)

  return cropped_image, cropped_boxes, cropped_classes


def normalize_image(image):
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(
      ssd_constants.NORMALIZATION_MEAN)[tf.newaxis, tf.newaxis, :]

  image /= tf.constant(
      ssd_constants.NORMALIZATION_STD)[tf.newaxis, tf.newaxis, :]

  return image


def encode_labels(boxes, classes):
  similarity_calc = region_similarity_calculator.IouSimilarity()
  matcher = argmax_matcher.ArgMaxMatcher(
      matched_threshold=ssd_constants.MATCH_THRESHOLD,
      unmatched_threshold=ssd_constants.MATCH_THRESHOLD,
      negatives_lower_than_unmatched=True,
      force_match_for_each_row=True)

  box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
      scale_factors=ssd_constants.BOX_CODER_SCALES)

  default_boxes = box_list.BoxList(tf.convert_to_tensor(DefaultBoxes()('ltrb')))
  target_boxes = box_list.BoxList(boxes)

  assigner = target_assigner.TargetAssigner(
      similarity_calc, matcher, box_coder)

  return assigner.assign(default_boxes, target_boxes, classes)


class SSDInputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, is_training):
    self._file_pattern = file_pattern
    self._is_training = is_training

  def __call__(self, params):
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _parse_example(data):
      with tf.name_scope('augmentation'):
        source_id = data['source_id']
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

        if self._is_training:
          image, boxes, classes = ssd_crop(image, boxes, classes)
          image, boxes = preprocessor.random_horizontal_flip(
              image=image, boxes=boxes)

          # TODO(someone in object detection): Color Jitter
          image = normalize_image(image)

          encoded_classes, _, encoded_boxes, _, _ = encode_labels(boxes,
                                                                  classes)

          # TODO(taylorrobie): Check that this cast is valid.
          # encoded_classes = tf.cast(encoded_classes, tf.int32)
          # labels = {
          #     ssd_constants.NUM_BOXES: tf.shape(boxes)[0],
          #     ssd_constants.BOXES: encoded_boxes,
          #     ssd_constants.CLASSES: encoded_classes,
          # }
          # TODO(haoyuzhang): measure or remove this overhead of concat
          # Encode labels (number of boxes, coordinates of bounding boxes, and
          # classes into a single tensor, in order to be compatible with
          # staging area in data input pipeline.
          # Step 1: pack box coordinates and classes together
          #     [nboxes, 4] concat [nboxes, 1] ==> [nboxes, 5]
          labels = tf.concat([encoded_boxes, encoded_classes], 1)
          # Step 2 (HACK): repeat number of boxes (a scalar tensor) five times,
          #                and pack result from Step 1 with it.
          #     [nboxes, 5] concat [1, 5] ==> [nboxes + 1, 5]
          nboxes = tf.shape(boxes)[0]                   # scalar, shape (,)
          nboxes_1d = tf.tile(tf.expand_dims(nboxes, 0), [5])
                                                        # 1D tensor, shape (5)
          nboxes_2d = tf.expand_dims(nboxes_1d, 0)      # 2D tensor, shape (1,5)
          labels = tf.concat([labels, tf.cast(nboxes_2d, tf.float32)], 0)
          return image, labels

        else:
          image = tf.image.resize_images(
              image[tf.newaxis, :, :, :],
              size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE)
          )[0, :, :, :]

          def trim_and_pad(inp_tensor):
            """Limit the number of boxes, and pad if necessary."""
            inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
            num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
            inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
            return tf.reshape(inp_tensor, [ssd_constants.MAX_NUM_EVAL_BOXES,
                                           inp_tensor.get_shape()[1]])

          boxes, classes = trim_and_pad(boxes), trim_and_pad(classes)
          return {
              ssd_constants.IMAGE: image,
              ssd_constants.BOXES: boxes,
              ssd_constants.CLASSES: classes,
              ssd_constants.SOURCE_ID: source_id,
          }

    batch_size_per_split = params['batch_size_per_split']
    num_splits = params['num_splits']
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._is_training)
    if self._is_training:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(batch_size_per_split)
      return dataset
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=self._is_training))

    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(example_decoder.decode, num_parallel_calls=64)

    # TODO(taylorrobie): Confirm that this is MLPerf rules compliant.
    dataset = dataset.filter(
        lambda data: tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0))
    dataset = dataset.apply(batching.map_and_batch(
        map_func=_parse_example,
        batch_size=batch_size_per_split,
        num_parallel_batches=num_splits,
        drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=num_splits)
    return dataset
