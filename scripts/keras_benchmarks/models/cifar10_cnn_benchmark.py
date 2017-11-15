'''
Original Model from keras/examples

Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model

from model import BenchmarkModel
from models import timehistory
if keras.backend.backend() == 'cntk':
    from gpu_mode import cntk_gpu_mode_config

from data_generator import generate_img_input_data
import numpy as np

class Cifar10CnnBenchmark(BenchmarkModel):

    # TODO(anjalisridhar): you can pass test name and sample type when creating
    # the object
    def __init__(self):
        self._test_name = "cifar10_cnn"
        self._sample_type="images"
        self._total_time = 0
        self._batch_size = 32
        self._epochs = 2
        self._num_samples = 1000

    def run_benchmark(self, gpus=0):
        num_classes = 10

        # Generate random input data
        input_shape = (self._num_samples, 3, 32, 32)
        x_train, y_train = generate_img_input_data(input_shape)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_train = keras.utils.to_categorical(y_train, 10)

        if keras.backend.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)


        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_train /= 255

        # create a distributed trainer for cntk
        if keras.backend.backend() is "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model, x_train.shape[0])
            x_train = x_train[start: end]
            y_train = y_train[start: end]

        start_time = time.time()
        time_callback = timehistory.TimeHistory()

        model.fit(x_train,
                  y_train,
                  batch_size=self._batch_size,
                  epochs=self._epochs,
                  shuffle=True,
                  callbacks=[time_callback])

        self._total_time = time.time() - start_time - time_callback.times[0]

    def get_totaltime(self):
        return self._total_time

    def get_iters(self):
        return self._epochs

    def get_testname(self):
        return self._test_name

    def get_sampletype(self):
        return self._sample_type

    def get_batch_size(self):
        return self._batch_size
