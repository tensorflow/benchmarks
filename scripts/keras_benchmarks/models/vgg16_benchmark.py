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
if keras.backend.backend() == 'cntk':
  from gpu_mode import cntk_gpu_mode_config


class VGG16Benchmark:

  def __init__(self):
    self.test_name = "vgg16"
    self.sample_type = "images"
    self.total_time = 0
    self.batch_size = 32
    self.epochs = 2
    self.num_samples = 1000

  def run_benchmark(self, gpus=0):
    input_shape = (128, 3, 224, 224)
    num_classes = 1000

    x_train, y_train = generate_img_input_data(input_shape, num_classes)

    model = applications.VGG16(weights=None)

    y_train = keras.utils.to_categorical(y_train, num_classes)

    if keras.backend.image_data_format() == 'channels_last':
      x_train = x_train.transpose(0, 2, 3, 1)

    x_train = x_train.astype('float32')
    x_train /= 255

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    if keras.backend.backend() is "tensorflow" and gpus > 1:
      model = multi_gpu_model(model, gpus=gpus)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # create a distributed trainer for cntk
    if keras.backend.backend() is "cntk" and gpus > 1:
      start, end = cntk_gpu_mode_config(model, x_train.shape[0])
      x_train = x_train[start: end]
      y_train = y_train[start: end]

    time_callback = timehistory.TimeHistory()

    model.fit(x_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        shuffle=True)

    self.total_time = 0
    for i in range(1, self.epochs):
      self.total_time += time_callback.times[i]

    keras.backend.clear_session()
