'''
Original Model from keras/examples/mnist_mlp.py

Benchmark a simple MLP model.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from models import timehistory
from data_generator import generate_img_input_data
if keras.backend.backend() == 'cntk':
    from gpu_mode import cntk_gpu_mode_config


class MnistMlpBenchmark():

    def __init__(self):
        self.test_name = "mnist_mlp"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 128
        self.epochs = 2
        self.num_samples = 1000

    def run_benchmark(self, gpus=0):
        num_classes = 10

        # Generate random input data
        input_shape = (self.num_samples, 28, 28)
        x_train, y_train = generate_img_input_data(input_shape)

        x_train = x_train.reshape(self.num_samples, 784)
        x_train = x_train.astype('float32')
        x_train /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        # create a distributed trainer for cntk
        if keras.backend.backend() is "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model, x_train.shape[0])
            x_train = x_train[start: end]
            y_train = y_train[start: end]

        time_callback = timehistory.TimeHistory()
        model.fit(x_train, y_train, batch_size=self.batch_size,
                  epochs=self.epochs, verbose=1, callbacks=[time_callback])

        self.total_time = 0
        for i in range(1, self.epochs):
            self.total_time += time_callback.times[i]
