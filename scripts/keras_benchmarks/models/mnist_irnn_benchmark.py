'''
Original Model from keras/examples

This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf
Optimizer is replaced with RMSprop which yields more stable and steady
improvement.
Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function

import keras
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop

from model import BenchmarkModel
from models import timehistory


class MnistIrnnBenchmark(BenchmarkModel):

  # TODO(anjalisridhar): you can pass test name and sample type when creating
  # the object
  def __init__(self):
    self._test_name = "mnist_irnn"
    self._sample_type="images"
    self._total_time = 0
    self._batch_size = 32
    self._epochs = 2

  def benchmarkMnistIrnn(self, keras_backend=None, gpu_count=0):
    if keras_backend is None:
      raise ValueError('keras_backend parameter must be specified.')

    if keras_backend is not "tensorflow" and gpu_count > 0:
      raise ValueError('gpu mode is currently only supported for tensorflow backends.')

    num_classes = 10
    hidden_units = 100

    learning_rate = 1e-6
    clip_norm = 1.0

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Evaluate IRNN...')
    model = Sequential()
    model.add(SimpleRNN(hidden_units,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        recurrent_initializer=initializers.Identity(gain=1.0),
                        activation='relu',
                        input_shape=x_train.shape[1:]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])


    start_time = time.time()
    time_callback = timehistory.TimeHistory()

    model.fit(x_train, y_train,
              batch_size=self._batch_size,
              epochs=self._epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[time_callback])

    self._total_time = time.time() - start_time - time_callback.times[0]

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('IRNN test score:', scores[0])
    print('IRNN test accuracy:', scores[1])

  def get_totaltime(self):
    return self._total_time

  def get_iters(self):
    return self._epochs - 1

  def get_testname(self):
    return self._test_name

  def get_sampletype(self):
    return self._sample_type

  def get_batch_size(self):
    return self._batch_size