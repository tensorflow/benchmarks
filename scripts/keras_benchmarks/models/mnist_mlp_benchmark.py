'''
Original Model from keras/examples

Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from model import BenchmarkModel
from models import timehistory
from ..gpu_mode import cntk_gpu_mode_config

class MnistMlpBenchmark(BenchmarkModel):

    # TODO(anjalisridhar): you can pass test name and sample type when creating
    # the object
    def __init__(self):
      self._test_name = "mnist_mlp"
      self._sample_type="images"
      self._total_time = 0
      self._batch_size = 128
      self._epochs = 2

    def benchmarkMnistMlp(self, keras_backend=None, gpu_count=0):
        if keras_backend is None:
          raise ValueError('keras_backend parameter must be specified.')

        num_classes = 10

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        if str(keras_backend) is "tensorflow" and gpu_count > 1:
            model = multi_gpu_model(model, gpus=gpu_count)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        # create a distributed trainer for cntk
        if str(keras_backend) is "cntk" and gpu_count > 1:
            start,end = cntk_gpu_mode_config(model, x_train.shape[0])
            x_train = x_train[start: end]
            y_train = y_train[start: end]

        start_time = time.time()
        time_callback = timehistory.TimeHistory()
        model.fit(x_train, y_train, batch_size=self._batch_size,
                  epochs=self._epochs, verbose=1,
                  validation_data=(x_test, y_test), callbacks=[time_callback])

        self._total_time = time.time() - start_time - time_callback.times[0]

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
