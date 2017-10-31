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
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
from model import BenchmarkModel
from models import timehistory


class Cifar10CnnBenchmark(BenchmarkModel):

    # TODO(anjalisridhar): you can pass test name and sample type when creating
    # the object
    def __init__(self):
      self._test_name = "cifar10_cnn"
      self._sample_type="images"
      self._total_time = 0
      self._batch_size = 32
      self._epochs = 2

    def benchmarkCifar10Cnn(self, keras_backend=None, gpu_count=0):
        if keras_backend is None:
          raise ValueError('keras_backend parameter must be specified.')

        num_classes = 10
        data_augmentation = True
        num_predictions = 20
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'keras_cifar10_trained_model.h5'

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

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

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        start_time = time.time()
        time_callback = timehistory.TimeHistory()

        model.fit(x_train, y_train,
                    batch_size=self._batch_size,
                    epochs=self._epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[time_callback])

        self._total_time = time.time() - start_time - time_callback.times[0]

        # Save model and weights
        if not os.path.isdir(save_dir):
          os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


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
