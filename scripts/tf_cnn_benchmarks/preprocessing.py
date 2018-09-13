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

"""Image pre-processing utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
import cnn_util
import data_utils
import ssd_constants


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


_RESIZE_METHOD_MAP = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}


def get_image_resize_method(resize_method, batch_position=0):
  """Get tensorflow resize method.

  If resize_method is 'round_robin', return different methods based on batch
  position in a round-robin fashion. NOTE: If the batch size is not a multiple
  of the number of methods, then the distribution of methods will not be
  uniform.

  Args:
    resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
    batch_position: position of the image in a batch. NOTE: this argument can
      be an integer or a tensor
  Returns:
    one of resize type defined in tf.image.ResizeMethod.
  """

  if resize_method != 'round_robin':
    return _RESIZE_METHOD_MAP[resize_method]

  # return a resize method based on batch position in a round-robin fashion.
  resize_methods = list(_RESIZE_METHOD_MAP.values())
  def lookup(index):
    return resize_methods[index]

  def resize_method_0():
    return utils.smart_cond(batch_position % len(resize_methods) == 0,
                            lambda: lookup(0), resize_method_1)

  def resize_method_1():
    return utils.smart_cond(batch_position % len(resize_methods) == 1,
                            lambda: lookup(1), resize_method_2)

  def resize_method_2():
    return utils.smart_cond(batch_position % len(resize_methods) == 2,
                            lambda: lookup(2), lambda: lookup(3))

  # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
  # because TF would not be able to construct a finite graph.

  return resize_method_0()


def decode_jpeg(image_buffer, scope=None):  # , dtype=tf.float32):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  # with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
  # with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
  with tf.name_scope(scope or 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 fancy_upscaling=False,
                                 dct_method='INTEGER_FAST')

    # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

    return image


def normalized_image(images):
  # Rescale from [0, 255] to [0, 2]
  images = tf.multiply(images, 1. / 127.5)
  # Rescale to [-1, 1]
  return tf.subtract(images, 1.0)


def eval_image(image,
               height,
               width,
               batch_position,
               resize_method,
               summary_verbosity=0):
  """Get the image for model evaluation.

  We preprocess the image simiarly to Slim, see
  https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
  Validation images do not have bounding boxes, so to crop the image, we first
  resize the image such that the aspect ratio is maintained and the resized
  height and width are both at least 1.15 times `height` and `width`
  respectively. Then, we do a central crop to size (`height`, `width`).

  Args:
    image: 3-D float Tensor representing the image.
    height: The height of the image that will be returned.
    width: The width of the image that will be returned.
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: one of the strings 'round_robin', 'nearest', 'bilinear',
      'bicubic', or 'area'.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
  Returns:
    An image of size (output_height, output_width, 3) that is resized and
    cropped as described above.
  """
  # TODO(reedwm): Currently we resize then crop. Investigate if it's faster to
  # crop then resize.
  with tf.name_scope('eval_image'):
    if summary_verbosity >= 3:
      tf.summary.image(
          'original_image', tf.expand_dims(image, 0))

    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    image_height_float = tf.cast(image_height, tf.float32)
    image_width_float = tf.cast(image_width, tf.float32)

    scale_factor = 1.15

    # Compute resize_height and resize_width to be the minimum values such that
    #   1. The aspect ratio is maintained (i.e. resize_height / resize_width is
    #      image_height / image_width), and
    #   2. resize_height >= height * `scale_factor`, and
    #   3. resize_width >= width * `scale_factor`
    max_ratio = tf.maximum(height / image_height_float,
                           width / image_width_float)
    resize_height = tf.cast(image_height_float * max_ratio * scale_factor,
                            tf.int32)
    resize_width = tf.cast(image_width_float * max_ratio * scale_factor,
                           tf.int32)

    # Resize the image to shape (`resize_height`, `resize_width`)
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(image,
                                             [resize_height, resize_width],
                                             image_resize_method,
                                             align_corners=False)

    # Do a central crop of the image to size (height, width).
    total_crop_height = (resize_height - height)
    crop_top = total_crop_height // 2
    total_crop_width = (resize_width - width)
    crop_left = total_crop_width // 2
    distorted_image = tf.slice(distorted_image, [crop_top, crop_left, 0],
                               [height, width, 3])

    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image(
          'cropped_resized_image', tf.expand_dims(distorted_image, 0))
    image = distorted_image
  return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: round_robin, nearest, bilinear, bicubic, or area.
    distortions: If true, apply full distortions for image colors.
    scope: Optional scope for op_scope.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
    distort_color_in_yiq: distort color of input images in YIQ space.
    fuse_decode_and_crop: fuse the decode/crop operation.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope(scope or 'distort_image'):
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if summary_verbosity >= 3:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image(
          'images_with_distorted_bounding_box',
          image_with_distorted_box)

    # Crop the image to the specified bounding box.
    if fuse_decode_and_crop:
      offset_y, offset_x, _ = tf.unstack(bbox_begin)
      target_height, target_width, _ = tf.unstack(bbox_size)
      crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
      image = tf.image.decode_and_crop_jpeg(
          image_buffer, crop_window, channels=3)
    else:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.slice(image, bbox_begin, bbox_size)

    distorted_image = tf.image.random_flip_left_right(image)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width],
        image_resize_method,
        align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image('cropped_resized_maybe_flipped_image',
                       tf.expand_dims(distorted_image, 0))

    if distortions:
      distorted_image = tf.cast(distorted_image, dtype=tf.float32)
      # Images values are expected to be in [0,1] for color distortion.
      distorted_image /= 255.
      # Randomly distort the colors.
      distorted_image = distort_color(distorted_image, batch_position,
                                      distort_color_in_yiq=distort_color_in_yiq)

      # Note: This ensures the scaling matches the output of eval_image
      distorted_image *= 255

    if summary_verbosity >= 3:
      tf.summary.image(
          'final_distorted_image',
          tf.expand_dims(distorted_image, 0))
    return distorted_image


def distort_color(image, batch_position=0, distort_color_in_yiq=False,
                  scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops based on the position of the image in a batch.

  Args:
    image: float32 Tensor containing single image. Tensor values should be in
      range [0, 1].
    batch_position: the position of the image in a batch. NOTE: this argument
      can be an integer or a tensor
    distort_color_in_yiq: distort color of input images in YIQ space.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(scope or 'distort_color'):

    def distort_fn_0(image=image):
      """Variant 0 of distort function."""
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      if distort_color_in_yiq:
        image = distort_image_ops.random_hsv_in_yiq(
            image, lower_saturation=0.5, upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      return image

    def distort_fn_1(image=image):
      """Variant 1 of distort function."""
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      if distort_color_in_yiq:
        image = distort_image_ops.random_hsv_in_yiq(
            image, lower_saturation=0.5, upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      return image

    image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                             distort_fn_1)
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


class InputPreprocessor(object):
  """Base class for all model preprocessors."""

  def __init__(self, batch_size, output_shape):
    self.batch_size = batch_size
    self.output_shape = output_shape


class BaseImagePreprocessor(InputPreprocessor):
  """Base class for all image model preprocessors."""

  def __init__(self,
               batch_size,
               output_shape,
               num_splits,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio=-1,
               summary_verbosity=0,
               distort_color_in_yiq=True,
               fuse_decode_and_crop=True):
    super(BaseImagePreprocessor, self).__init__(batch_size, output_shape)
    # output_shape is in form (height, width, depth)
    self.height = output_shape[0]
    self.width = output_shape[1]
    self.depth = output_shape[2]
    self.num_splits = num_splits
    self.dtype = dtype
    self.train = train
    self.resize_method = resize_method
    self.shift_ratio = shift_ratio
    self.distortions = distortions
    self.distort_color_in_yiq = distort_color_in_yiq
    self.fuse_decode_and_crop = fuse_decode_and_crop
    if self.batch_size % self.num_splits != 0:
      raise ValueError(
          ('batch_size must be a multiple of num_splits: '
           'batch_size %d, num_splits: %d') %
          (self.batch_size, self.num_splits))
    self.batch_size_per_split = self.batch_size // self.num_splits
    self.summary_verbosity = summary_verbosity

  def preprocess(self, image_buffer, bbox, batch_position):
    raise NotImplementedError('Must be implemented by subclass.')

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio):
    raise NotImplementedError('Must be implemented by subclass.')

  def supports_datasets(self):
    return False


class RecordInputImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor for images with RecordInput format."""

  def preprocess(self, image_buffer, bbox, batch_position):
    """Preprocessing image_buffer as a function of its batch position."""
    if self.train:
      image = train_image(image_buffer, self.height, self.width, bbox,
                          batch_position, self.resize_method, self.distortions,
                          None, summary_verbosity=self.summary_verbosity,
                          distort_color_in_yiq=self.distort_color_in_yiq,
                          fuse_decode_and_crop=self.fuse_decode_and_crop)
    else:
      image = tf.image.decode_jpeg(
          image_buffer, channels=3, dct_method='INTEGER_FAST')
      image = eval_image(image, self.height, self.width, batch_position,
                         self.resize_method,
                         summary_verbosity=self.summary_verbosity)
    # Note: image is now float32 [height,width,3] with range [0, 255]

    # image = tf.cast(image, tf.uint8) # HACK TESTING

    normalized = normalized_image(image)
    return tf.cast(normalized, self.dtype)

  def parse_and_preprocess(self, value, batch_position):
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    image = self.preprocess(image_buffer, bbox, batch_position)
    return (label_index, image)

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    if shift_ratio < 0:
      shift_ratio = self.shift_ratio
    with tf.name_scope('batch_processing'):
      # Build final results per split.
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]
      if use_datasets:
        ds = data_utils.create_dataset(
            self.batch_size, self.num_splits, self.batch_size_per_split,
            self.parse_and_preprocess, dataset, subset, self.train, cache_data)
        ds_iterator = data_utils.create_iterator(ds)
        for d in xrange(self.num_splits):
          labels[d], images[d] = ds_iterator.get_next()

      else:
        record_input = data_flow_ops.RecordInput(
            file_pattern=dataset.tf_record_pattern(subset),
            seed=301,
            parallelism=64,
            buffer_size=10000,
            batch_size=self.batch_size,
            shift_ratio=shift_ratio,
            name='record_input')
        records = record_input.get_yield_op()
        records = tf.split(records, self.batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        for idx in xrange(self.batch_size):
          value = records[idx]
          (label, image) = self.parse_and_preprocess(value, idx)
          split_index = idx % self.num_splits
          labels[split_index].append(label)
          images[split_index].append(image)

      for split_index in xrange(self.num_splits):
        if not use_datasets:
          images[split_index] = tf.parallel_stack(images[split_index])
          labels[split_index] = tf.concat(labels[split_index], 0)
        images[split_index] = tf.reshape(
            images[split_index],
            shape=[self.batch_size_per_split, self.height, self.width,
                   self.depth])
        labels[split_index] = tf.reshape(labels[split_index],
                                         [self.batch_size_per_split])
      return images, labels

  def supports_datasets(self):
    return True


class ImagenetPreprocessor(RecordInputImagePreprocessor):

  def preprocess(self, image_buffer, bbox, batch_position):
    # pylint: disable=g-import-not-at-top
    try:
      from official.resnet.imagenet_preprocessing import preprocess_image
    except ImportError:
      tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH.')
      raise
    if self.train:
      image = preprocess_image(
          image_buffer, bbox, self.height, self.width, self.depth,
          is_training=True)
    else:
      image = preprocess_image(
          image_buffer, bbox, self.height, self.width, self.depth,
          is_training=False)
    return tf.cast(image, self.dtype)


class Cifar10ImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor for Cifar10 input images."""

  def _distort_image(self, image):
    """Distort one image for training a network.

    Adopted the standard data augmentation scheme that is widely used for
    this dataset: the images are first zero-padded with 4 pixels on each side,
    then randomly cropped to again produce distorted images; half of the images
    are then horizontally mirrored.

    Args:
      image: input image.
    Returns:
      distorted image.
    """
    image = tf.image.resize_image_with_crop_or_pad(
        image, self.height + 8, self.width + 8)
    distorted_image = tf.random_crop(image,
                                     [self.height, self.width, self.depth])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    if self.summary_verbosity >= 3:
      tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def _eval_image(self, image):
    """Get the image for model evaluation."""
    distorted_image = tf.image.resize_image_with_crop_or_pad(
        image, self.width, self.height)
    if self.summary_verbosity >= 3:
      tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def preprocess(self, raw_image):
    """Preprocessing raw image."""
    if self.summary_verbosity >= 3:
      tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
    if self.train and self.distortions:
      image = self._distort_image(raw_image)
    else:
      image = self._eval_image(raw_image)
    normalized = normalized_image(image)
    return tf.cast(normalized, self.dtype)

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    # TODO(jsimsa): Implement datasets code path
    del use_datasets, cache_data, shift_ratio
    with tf.name_scope('batch_processing'):
      all_images, all_labels = dataset.read_data_files(subset)
      all_images = tf.constant(all_images)
      all_labels = tf.constant(all_labels)
      input_image, input_label = tf.train.slice_input_producer(
          [all_images, all_labels])
      input_image = tf.cast(input_image, self.dtype)
      input_label = tf.cast(input_label, tf.int32)
      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                               min_fraction_of_examples_in_queue)
      raw_images, raw_labels = tf.train.shuffle_batch(
          [input_image, input_label], batch_size=self.batch_size,
          capacity=min_queue_examples + 3 * self.batch_size,
          min_after_dequeue=min_queue_examples)

      images = [[] for i in range(self.num_splits)]
      labels = [[] for i in range(self.num_splits)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(raw_images, axis=0)
      raw_labels = tf.unstack(raw_labels, axis=0)
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        # The raw image read from data has the format [depth, height, width]
        # reshape to the format returned by minibatch.
        raw_image = tf.reshape(raw_images[i],
                               [dataset.depth, dataset.height, dataset.width])
        raw_image = tf.transpose(raw_image, [1, 2, 0])
        image = self.preprocess(raw_image)
        images[split_index].append(image)

        labels[split_index].append(raw_labels[i])

      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])
      return images, labels


class COCOPreprocessor(BaseImagePreprocessor):
  """Preprocessor for COCO dataset input images, boxes, and labels."""

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    try:
      from object_detection.data_decoders import tf_example_decoder  # pylint: disable=g-import-not-at-top
      import ssd_dataloader  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs by '
                        'following https://github.com/tensorflow/models/blob/'
                        'master/research/object_detection/g3doc/installation.md'
                        '#protobuf-compilation')

    if shift_ratio < 0:
      shift_ratio = self.shift_ratio
    self.example_decoder = tf_example_decoder.TfExampleDecoder()

    with tf.name_scope('batch_processing'):
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]

      glob_pattern = dataset.tf_record_pattern(subset)
      filenames = gfile.Glob(glob_pattern)
      ssd_input = ssd_dataloader.SSDInputReader(filenames, subset == 'train')
      ds = ssd_input({'batch_size_per_split': self.batch_size_per_split,
                      'num_splits': self.num_splits})
      ds_iterator = data_utils.create_iterator(ds)
      for d in range(self.num_splits):
        images[d], labels[d] = ds_iterator.get_next()
      for split_index in xrange(self.num_splits):
        images[split_index] = tf.reshape(
            images[split_index],
            shape=[self.batch_size_per_split, self.height, self.width,
                   self.depth])
        labels[split_index] = tf.reshape(
            labels[split_index],
            # Encoded tensor with object category, number of bounding boxes and
            # their locations. The 0th dimension is batch size, and for each
            # item in batch the tensor looks like this ((x, y, w, h) is the
            # cordinates and size of a bounding box, c is class):
            #
            # [[x,      y,      w,      h,      c     ],       ^
            #  [x,      y,      w,      h,      c     ],       |
            #  [...,    ...,    ...,    ...,    ...   ],  NUM_SSD_BOXES+1
            #  [x,      y,      w,      h,      c     ],       |
            #  [nboxes, nboxes, nboxes, nboxes, nboxes]]       v
            #
            # |<---------- 4 cordinates + 1 ---------->|
            shape=[self.batch_size_per_split, ssd_constants.NUM_SSD_BOXES+1, 5])
      return images, labels


class SyntheticImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor used for images and labels."""

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    """Get synthetic cpu image batches."""
    raise NotImplementedError(
        'The SyntheticImagePreprocessor does not support getting images with '
        'minibatch().')


class TestImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor used for testing.

  set_fake_data() sets which images and labels will be output by minibatch(),
  and must be called before minibatch(). This allows tests to easily specify
  a set of images to use for training, without having to create any files.

  Queue runners must be started for this preprocessor to work.
  """

  def __init__(self,
               batch_size,
               output_shape,
               num_splits,
               dtype,
               train=None,
               distortions=None,
               resize_method=None,
               shift_ratio=0,
               summary_verbosity=0,
               distort_color_in_yiq=False,
               fuse_decode_and_crop=False):
    super(TestImagePreprocessor, self).__init__(
        batch_size, output_shape, num_splits, dtype, train, distortions,
        resize_method, shift_ratio, summary_verbosity=summary_verbosity,
        distort_color_in_yiq=distort_color_in_yiq,
        fuse_decode_and_crop=fuse_decode_and_crop)
    self.expected_subset = None

  def set_fake_data(self, fake_images, fake_labels):
    assert len(fake_images.shape) == 4
    assert len(fake_labels.shape) == 1
    num_images = fake_images.shape[0]
    assert num_images == fake_labels.shape[0]
    assert num_images % self.batch_size == 0
    self.fake_images = fake_images
    self.fake_labels = fake_labels

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=0):
    """Get test image batches."""
    del dataset, use_datasets, cache_data
    if (not hasattr(self, 'fake_images') or
        not hasattr(self, 'fake_labels')):
      raise ValueError('Must call set_fake_data() before calling minibatch '
                       'on TestImagePreprocessor')
    if self.expected_subset is not None:
      assert subset == self.expected_subset

    shift_ratio = shift_ratio or self.shift_ratio
    fake_images = cnn_util.roll_numpy_batches(self.fake_images, self.batch_size,
                                              shift_ratio)
    fake_labels = cnn_util.roll_numpy_batches(self.fake_labels, self.batch_size,
                                              shift_ratio)

    with tf.name_scope('batch_processing'):
      image_slice, label_slice = tf.train.slice_input_producer(
          [fake_images, fake_labels],
          shuffle=False,
          name='image_slice')
      raw_images, raw_labels = tf.train.batch(
          [image_slice, label_slice], batch_size=self.batch_size,
          name='image_batch')
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        raw_image = tf.cast(raw_images[i], self.dtype)
        images[split_index].append(raw_image)
        labels[split_index].append(raw_labels[i])
      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])

      normalized = normalized_image(images)
      return tf.cast(normalized, self.dtype), labels
