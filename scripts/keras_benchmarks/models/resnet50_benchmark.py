'''
Model constructed using keras/applications.

Benchmark a Resnet50 model.
'''

from __future__ import print_function

import keras
from keras import applications
from keras.utils import multi_gpu_model
from models import timehistory
from data_generator import generate_img_input_data
if keras.backend.backend() == 'cntk':
  from gpu_mode import cntk_gpu_mode_config, finalize

if keras.backend.backend() == 'tensorflow':
  import tensorflow as tf

def crossentropy_from_logits(y_true, y_pred):
  return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

class Resnet50Benchmark:

  def __init__(self):
    self.test_name = "resnet50"
    self.sample_type = "images"
    self.total_time = 0
    self.batch_size = 16
    self.epochs = 4
    self.num_samples = 1000
    self.test_type = 'channels_first, batchnorm, learning_phase, batch_size'

    self.use_dataset_tensors = False
    self.use_logits_loss = True

  def run_benchmark(self, gpus=0):
    print("Running model ", self.test_name)

    input_shape = (self.num_samples, 3, 224, 224)
    num_classes = 1000

    x_train, y_train = generate_img_input_data(input_shape, num_classes)

    if keras.backend.backend == "tensorflow" and gpus > 1:
      keras.backend.set_image_data_format('channels_first')

    keras.backend.set_learning_phase(True)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    if keras.backend.image_data_format() == 'channels_last':
      x_train = x_train.transpose(0, 2, 3, 1)

    x_train /= 255

    input_tensor = keras.layers.Input(tensor=keras.backend.constant(x_train))
    base_model = applications.ResNet50(input_tensor=input_tensor, include_top=False, pooling='avg', weights=None)
    output = base_model.output
    predictions = keras.layers.Dense(num_classes)(output)
    model = keras.models.Model(input_tensor, predictions)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                  loss=crossentropy_from_logits,
                  metrics=['accuracy'])

    time_callback = timehistory.TimeHistory()

    model.fit(y=y_train, batch_size=self.batch_size,
              epochs=self.epochs, shuffle=True, callbacks=[time_callback])

    self.total_time = 0
    for i in range(1, self.epochs):
      self.total_time += time_callback.times[i]

    if keras.backend.backend() == "tensorflow":
      keras.backend.clear_session()

    if keras.backend.backend() == "cntk" and gpus > 1:
      finalize()

