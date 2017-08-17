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
import glob
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
import cnn_util

FLAGS = tf.flags.FLAGS


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


def get_image_resize_method(resize_method, thread_id=0):
  """Get tensorflow resize method.

  If method is 'round_robin', return different methods for different threads
  based on round-robin fashion.

  Args:
    resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
    thread_id: an integer id for the thread.
  Returns:
    one of resize type defined in tf.image.ResizeMethod.
  """
  resize_methods_map = {
      'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      'bilinear': tf.image.ResizeMethod.BILINEAR,
      'bicubic': tf.image.ResizeMethod.BICUBIC,
      'area': tf.image.ResizeMethod.AREA
  }

  if resize_method == 'round_robin':
    # return one of resize methods based round-robin fashion.
    resize_methods = resize_methods_map.values()
    return resize_methods[thread_id % len(resize_methods)]

  return resize_methods_map[resize_method]


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


def eval_image(image, height, width, bbox, thread_id, resize_method):
  """Get the image for model evaluation."""
  with tf.name_scope('eval_image'):
    if not thread_id and FLAGS.summary_verbosity >= 2:
      tf.summary.image(
          'original_image', tf.expand_dims(image, 0))

    if resize_method == 'crop':
      # Note: This is much slower than crop_to_bounding_box
      #         It seems that the redundant pad step has huge overhead
      # distorted_image = tf.image.resize_image_with_crop_or_pad(image,
      #                                                         height, width)
      shape = tf.shape(image)
      y0 = (shape[0] - height) // 2
      x0 = (shape[1] - width) // 2
      # distorted_image = tf.slice(image, [y0,x0,0], [height,width,3])
      distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height,
                                                      width)
    else:
      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bbox,
          min_object_covered=0.1,
          aspect_ratio_range=[0.75, 1.33],
          area_range=[0.05, 1.0],
          max_attempts=100,
          use_image_if_no_bounding_boxes=True)
      bbox_begin, bbox_size, _ = sample_distorted_bounding_box
      # Crop the image to the specified bounding box.
      distorted_image = tf.slice(image, bbox_begin, bbox_size)
      # TODO(reedwm): revise this resize method for eval.
      image_resize_method = get_image_resize_method(resize_method, thread_id)
      # This resizing operation may distort the images because the aspect
      # ratio is not respected.
      if cnn_util.tensorflow_version() >= 11:
        distorted_image = tf.image.resize_images(
            distorted_image, [height, width],
            image_resize_method,
            align_corners=False)
      else:
        distorted_image = tf.image.resize_images(
            distorted_image,
            height,
            width,
            image_resize_method,
            align_corners=False)
    distorted_image.set_shape([height, width, 3])
    if not thread_id and FLAGS.summary_verbosity >= 2:
      tf.summary.image(
          'cropped_resized_image', tf.expand_dims(distorted_image, 0))
    image = distorted_image
  return image


def train_image(image,
                height,
                width,
                bbox,
                thread_id,
                resize_method,
                distortions,
                scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    resize_method: round_robin, nearest, bilinear, bicubic, or area.
    distortions: If true, apply full distortions for image colors.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope(scope or 'distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    if distortions:
      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      # Display the bounding box in the first thread only.
      if not thread_id and FLAGS.summary_verbosity >= 2:
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox)
        tf.summary.image(
            'image_with_bounding_boxes', image_with_box)

    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if FLAGS.summary_verbosity >= 2:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image(
          'images_with_distorted_bounding_box',
          image_with_distorted_box)

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    image_resize_method = get_image_resize_method(resize_method, thread_id)
    if cnn_util.tensorflow_version() >= 11:
      distorted_image = tf.image.resize_images(
          distorted_image, [height, width],
          image_resize_method,
          align_corners=False)
    else:
      distorted_image = tf.image.resize_images(
          distorted_image,
          height,
          width,
          image_resize_method,
          align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if FLAGS.summary_verbosity >= 2:
      tf.summary.image(
          'cropped_resized_image',
          tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    if distortions:
      # Randomly distort the colors.
      distorted_image = distort_color(distorted_image, thread_id)

      # Note: This ensures the scaling matches the output of eval_image
      distorted_image *= 256

    if FLAGS.summary_verbosity >= 2:
      tf.summary.image(
          'final_distorted_image',
          tf.expand_dims(distorted_image, 0))
    return distorted_image


def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  # with tf.op_scope([image], scope, 'distort_color'):
  # with tf.name_scope(scope, 'distort_color', [image]):
  with tf.name_scope(scope or 'distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


class RecordInputImagePreprocessor(object):
  """Preprocessor for images with RecordInput format."""

  def __init__(self,
               height,
               width,
               batch_size,
               device_count,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio):
    self.height = height
    self.width = width
    self.batch_size = batch_size
    self.device_count = device_count
    self.dtype = dtype
    self.train = train
    self.resize_method = resize_method
    self.shift_ratio = shift_ratio
    self.distortions = distortions
    if self.batch_size % self.device_count != 0:
      raise ValueError(
          ('batch_size must be a multiple of device_count: '
           'batch_size %d, device_count: %d') %
          (self.batch_size, self.device_count))
    self.batch_size_per_device = self.batch_size // self.device_count

  def preprocess(self, image_buffer, bbox, thread_id):
    """Preprocessing image_buffer using thread_id."""
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 dct_method='INTEGER_FAST')
    if self.train:
      image = train_image(image, self.height, self.width, bbox, thread_id,
                          self.resize_method, self.distortions)
    else:
      image = eval_image(image, self.height, self.width, bbox, thread_id,
                         self.resize_method)
    # Note: image is now float32 [height,width,3] with range [0, 255]

    # image = tf.cast(image, tf.uint8) # HACK TESTING

    return image

  def parse_and_preprocess(self, value, counter):
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    image = self.preprocess(image_buffer, bbox, counter % 4)
    return (label_index, image)

  def minibatch(self, dataset, subset, use_data_sets):
    with tf.name_scope('batch_processing'):
      images = [[] for i in range(self.device_count)]
      labels = [[] for i in range(self.device_count)]
      if use_data_sets:
        file_names = glob.glob(dataset.tf_record_pattern(subset))
        batch_size_per = self.batch_size / self.device_count
        num_threads = 10
        output_buffer_size = num_threads * 2000

        counter = tf.contrib.data.Dataset.range(sys.maxint)
        ds = tf.contrib.data.TFRecordDataset(file_names)
        ds = tf.contrib.data.Dataset.zip((ds, counter))
        ds = ds.map(
            self.parse_and_preprocess,
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
        shuffle_buffer_size = 10000
        ds = ds.shuffle(shuffle_buffer_size)
        repeat_count = -1  # infinite repetition
        ds = ds.repeat(repeat_count)
        ds = ds.batch(batch_size_per)
        ds_iterator = ds.make_one_shot_iterator()

        for d in xrange(self.device_count):
          labels[d], images[d] = ds_iterator.get_next()

      else:
        # Build final results per device.
        record_input = data_flow_ops.RecordInput(
            file_pattern=dataset.tf_record_pattern(subset),
            seed=301,
            parallelism=64,
            buffer_size=10000,
            batch_size=self.batch_size,
            shift_ratio=self.shift_ratio,
            name='record_input')
        records = record_input.get_yield_op()
        records = tf.split(records, self.batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        for i in xrange(self.batch_size):
          value = records[i]
          (label_index, image) = self.parse_and_preprocess(value, i % 4)
          device_index = i % self.device_count
          images[device_index].append(image)
          labels[device_index].append(label_index)

      label_index_batch = [None] * self.device_count
      for device_index in xrange(self.device_count):
        if use_data_sets:
          label_index_batch[device_index] = labels[device_index]
        else:
          images[device_index] = tf.parallel_stack(images[device_index])
          label_index_batch[device_index] = tf.concat(labels[device_index], 0)
        images[device_index] = tf.cast(images[device_index], self.dtype)
        depth = 3
        images[device_index] = tf.reshape(
            images[device_index],
            shape=[self.batch_size_per_device, self.height, self.width, depth])
        label_index_batch[device_index] = tf.reshape(
            label_index_batch[device_index], [self.batch_size_per_device])
        if FLAGS.summary_verbosity >= 2:
          # Display the training images in the visualizer.
          tf.summary.image('images', images)

      return images, label_index_batch


class Cifar10ImagePreprocessor(object):
  """Preprocessor for Cifar10 input images."""

  def __init__(self,
               height,
               width,
               batch_size,
               device_count,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio):
    # Process images of this size. Depending on the model configuration, the
    # size of the input layer might differ from the original size of 32 x 32.
    self.height = height or 32
    self.width = width or 32
    self.depth = 3
    self.batch_size = batch_size
    self.device_count = device_count
    self.dtype = dtype
    self.train = train
    self.distortions = distortions
    del resize_method
    del shift_ratio  # unused, because a RecordInput is not used
    if self.batch_size % self.device_count != 0:
      raise ValueError(
          ('batch_size must be a multiple of device_count: '
           'batch_size %d, device_count: %d') %
          (self.batch_size, self.device_count))
    self.batch_size_per_device = self.batch_size // self.device_count

  def _distort_image(self, image):
    """Distort one image for training a network.

    Adopted the standard data augmentation scheme that is widely used for
    this dataset: the images are first zero-padded with 4 pixels on each side,
    then randomly cropped to again produce distorted images; half of the images
    are then horizontally mirrored.

    Args:
      image: input image.
    Returns:
      distored image.
    """
    image = tf.image.resize_image_with_crop_or_pad(
        image, self.height + 8, self.width + 8)
    distorted_image = tf.random_crop(image,
                                     [self.height, self.width, self.depth])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    if FLAGS.summary_verbosity >= 2:
      tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def _eval_image(self, image):
    """Get the image for model evaluation."""
    distorted_image = tf.image.resize_image_with_crop_or_pad(
        image, self.width, self.height)
    if FLAGS.summary_verbosity >= 2:
      tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def preprocess(self, raw_image):
    """Preprocessing raw image."""
    if FLAGS.summary_verbosity >= 2:
      tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
    if self.train and self.distortions:
      image = self._distort_image(raw_image)
    else:
      image = self._eval_image(raw_image)
    return image

  def minibatch(self, dataset, subset, use_data_sets):
    # TODO(jsimsa): Implement data sets code path
    del use_data_sets
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

      images = [[] for i in range(self.device_count)]
      labels = [[] for i in range(self.device_count)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(raw_images, axis=0)
      raw_labels = tf.unstack(raw_labels, axis=0)
      for i in xrange(self.batch_size):
        device_index = i % self.device_count
        # The raw image read from data has the format [depth, height, width]
        # reshape to the format returned by minibatch.
        raw_image = tf.reshape(raw_images[i],
                               [dataset.depth, dataset.height, dataset.width])
        raw_image = tf.transpose(raw_image, [1, 2, 0])
        image = self.preprocess(raw_image)
        images[device_index].append(image)

        labels[device_index].append(raw_labels[i])

      for device_index in xrange(self.device_count):
        images[device_index] = tf.parallel_stack(images[device_index])
        labels[device_index] = tf.parallel_stack(labels[device_index])
      return images, labels


class TestImagePreprocessor(object):
  """Preprocessor used for testing.

  set_fake_data() sets which images and labels will be output by minibatch(),
  and must be called before minibatch(). This allows tests to easily specify
  a set of images to use for training, without having to create any files.

  Queue runners must be started for this preprocessor to work.
  """

  def __init__(self,
               height,
               width,
               batch_size,
               device_count,
               dtype=None,
               train=None,
               distortions=None,
               resize_method=None,
               shift_ratio=0):
    del height, width, dtype, train, distortions, resize_method, shift_ratio
    self.batch_size = batch_size
    self.device_count = device_count
    self.expected_subset = None

  def set_fake_data(self, fake_images, fake_labels):
    assert len(fake_images.shape) == 4
    assert len(fake_labels.shape) == 1
    assert fake_images.shape[0] == fake_labels.shape[0]
    assert fake_images.shape[0] % self.batch_size == 0
    self.fake_images = fake_images
    self.fake_labels = fake_labels

  def minibatch(self, dataset, subset, use_data_sets):
    del dataset, use_data_sets
    if (not hasattr(self, 'fake_images') or
        not hasattr(self, 'fake_labels')):
      raise ValueError('Must call set_fake_data() before calling minibatch '
                       'on TestImagePreprocessor')
    if self.expected_subset is not None:
      assert subset == self.expected_subset

    with tf.name_scope('batch_processing'):
      image_slice = tf.train.slice_input_producer(
          [self.fake_images],
          shuffle=False,
          name='image_slice')
      raw_images = tf.train.batch(image_slice, self.batch_size,
                                  name='image_batch')
      label_slice = tf.train.slice_input_producer(
          [self.fake_labels],
          shuffle=False,
          name='label_slice')
      raw_labels = tf.train.batch(label_slice, self.batch_size,
                                  name='label_batch')

      images = [[] for _ in range(self.device_count)]
      labels = [[] for _ in range(self.device_count)]
      for i in xrange(self.batch_size):
        device_index = i % self.device_count
        images[device_index].append(raw_images[i])
        labels[device_index].append(raw_labels[i])
      for device_index in xrange(self.device_count):
        images[device_index] = tf.parallel_stack(images[device_index])
        labels[device_index] = tf.parallel_stack(labels[device_index])

      return images, labels
