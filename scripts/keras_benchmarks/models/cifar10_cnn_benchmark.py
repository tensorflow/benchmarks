'''
Original Model from keras/examples/cifar10_cnn.py

Benchmark CNN model
'''

from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model

from models import timehistory
from data_generator import generate_img_input_data
if keras.backend.backend() == 'cntk':
    from gpu_mode import cntk_gpu_mode_config


class Cifar10CnnBenchmark():

    def __init__(self):
        self.test_name = "cifar10_cnn"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 32
        self.epochs = 2
        self.num_samples = 1000

    def run_benchmark(self, gpus=0):
        num_classes = 10

        # Generate random input data
        input_shape = (self.num_samples, 3, 32, 32)
        x_train, y_train = generate_img_input_data(input_shape)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_train = keras.utils.to_categorical(y_train, 10)

        if keras.backend.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

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

        time_callback = timehistory.TimeHistory()

        model.fit(x_train,
                  y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  shuffle=True,
                  callbacks=[time_callback])

        self.total_time = 0
        for i in range(1, self.epochs):
            self.total_time += time_callback.times[i]
