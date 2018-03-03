'''
Model constructed using keras/applications.

Benchmark a VGG16 model.
'''

from __future__ import print_function

import keras
from keras import applications
from keras.utils import multi_gpu_model

from models import timehistory
from data_generator import generate_img_input_data
if keras.backend.backend() == 'tensorflow':
  import tensorflow as tf
if keras.backend.backend() == 'cntk':
  from gpu_mode import cntk_gpu_mode_config, finalize

def crossentropy_from_logits(y_true, y_pred):
  return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

class VGG16Benchmark:

  def __init__(self):
    self.test_name = "vgg16"
    self.sample_type = "images"
    self.total_time = 0
    self.batch_size = 8
    self.epochs = 4
    self.num_samples = 1000

  def run_benchmark(self, gpus=0, use_dataset_tensors=False):
    print("Running model ", self.test_name)
    keras.backend.set_learning_phase(True)

    input_shape = (self.num_samples, 3, 224, 224)
    num_classes = 1000

    x_train, y_train = generate_img_input_data(input_shape, num_classes)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    if keras.backend.backend == "tensorflow" and gpus > 1:
      keras.backend.set_image_data_format('channels_first')

    if keras.backend.image_data_format() == 'channels_last':
      x_train = x_train.transpose(0, 2, 3, 1)
    x_train /= 255
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    if use_dataset_tensors:
      # Create the dataset and its associated one-shot iterator.
      dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
      dataset = dataset.repeat()
      dataset = dataset.shuffle(10000)
      dataset = dataset.batch(self.batch_size)
      iterator = dataset.make_one_shot_iterator()

      # Model creation using tensors from the get_next() graph node.
      inputs, targets = iterator.get_next()

    if use_dataset_tensors:
      input_tensor = keras.layers.Input(tensor=inputs)
      outputs = applications.ResNet50(include_top=False,
                                      pooling='avg',
                                      weights=None)(input_tensor)
      predictions = keras.layers.Dense(num_classes)(outputs)
      model = keras.models.Model(input_tensor, predictions)
    else:
      model = applications.VGG16(weights=None)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    if keras.backend.backend() == "tensorflow" and gpus > 1:
      model = multi_gpu_model(model, gpus=gpus)

    if use_dataset_tensors:
      model.compile(loss=crossentropy_from_logits,
                    optimizer=opt,
                    metrics=['accuracy'],
                    target_tensors=[targets])
    else:
      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    # create a distributed trainer for cntk
    if keras.backend.backend() == "cntk" and gpus > 1:
      start, end = cntk_gpu_mode_config(model, x_train.shape[0])
      x_train = x_train[start: end]
      y_train = y_train[start: end]

    time_callback = timehistory.TimeHistory()
    if use_dataset_tensors:
      model.fit(epochs=self.epochs, steps_per_epoch=15, callbacks=[time_callback])
    else:
      model.fit(x_train,
                          y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          shuffle=True,
                          callbacks=[time_callback])

    self.total_time = 0
    for i in range(1, self.epochs):
      self.total_time += time_callback.times[i]

    if keras.backend.backend() == "tensorflow":
      keras.backend.clear_session()

    if keras.backend.backend() == "cntk" and gpus > 1:
      finalize()
