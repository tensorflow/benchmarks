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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from model import BenchmarkModel
from models import timehistory
if keras.backend.backend() == 'cntk':
    from gpu_mode import cntk_gpu_mode_config
from data_generator import generate_img_input_data

class MnistIrnnBenchmark(BenchmarkModel):

    def __init__(self):
        self._test_name = "mnist_irnn"
        self._sample_type="images"
        self._total_time = 0
        self._batch_size = 32
        self._epochs = 2
        self._num_samples = 1000

    def run_benchmark(self, gpus=0):
        num_classes = 10
        hidden_units = 100
        learning_rate = 1e-6

        # Generate random input data
        input_shape = (self._num_samples, 28, 28)
        x_train, y_train = generate_img_input_data(input_shape)

        x_train = x_train.reshape(x_train.shape[0], -1, 1)
        x_train = x_train.astype('float32')
        x_train /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)

        model = Sequential()
        model.add(SimpleRNN(hidden_units,
                            kernel_initializer=initializers.RandomNormal(stddev=0.001),
                            recurrent_initializer=initializers.Identity(gain=1.0),
                            activation='relu',
                            input_shape=x_train.shape[1:]))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        rmsprop = RMSprop(lr=learning_rate)

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        model.compile(loss='categorical_crossentropy',
                      optimizer=rmsprop,
                      metrics=['accuracy'])

        # create a distributed trainer for cntk
        if keras.backend.backend() is "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model, x_train.shape[0])
            x_train = x_train[start: end]
            y_train = y_train[start: end]

        start_time = time.time()
        time_callback = timehistory.TimeHistory()

        model.fit(x_train, y_train,
                  batch_size=self._batch_size,
                  epochs=self._epochs,
                  verbose=1,
                  callbacks=[time_callback])

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
