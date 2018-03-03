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
    from gpu_mode import cntk_gpu_mode_config, finalize

if keras.backend.backend() == 'tensorflow':
  import tensorflow as tf

def crossentropy_from_logits(y_true, y_pred):
  return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

class MnistMlpBenchmark:

    def __init__(self):
        self.test_name = "mnist_mlp"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 32
        self.epochs = 4
        self.num_samples = 1000

    def run_benchmark(self, gpus=0, use_dataset_tensors=False):
        print("Running model ", self.test_name)
        keras.backend.set_learning_phase(True)

        num_classes = 10

        # Generate random input data
        input_shape = (self.num_samples, 28, 28)
        x_train, y_train = generate_img_input_data(input_shape, num_classes)
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        x_train = x_train.reshape(self.num_samples, 784)

        x_train /= 255
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

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
          model.add(Dense(num_classes))
          predictions = model(input_tensor)
          model = keras.models.Model(input_tensor, predictions)
        else:
          model.add(Dense(num_classes, activation='softmax'))

        if keras.backend.backend() == "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        if use_dataset_tensors:
          model.compile(loss=crossentropy_from_logits,
                        optimizer=RMSprop(),
                        metrics=['accuracy'],
                        target_tensors=[targets])
        else:
          model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        # create a distributed trainer for cntk
        if keras.backend.backend() == "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model.model, x_train.shape[0])
            x_train = x_train[start: end]
            y_train = y_train[start: end]

        time_callback = timehistory.TimeHistory()
        if use_dataset_tensors:
          model.fit(epochs=self.epochs, steps_per_epoch=15, callbacks=[time_callback])
        else:
          model.fit(x_train, y_train, batch_size=self.batch_size,
                        epochs=self.epochs, verbose=1, callbacks=[time_callback])

        self.total_time = 0
        for i in range(1, self.epochs):
            self.total_time += time_callback.times[i]

        if keras.backend.backend() == "tensorflow":
            keras.backend.clear_session()

        if keras.backend.backend() == "cntk" and gpus > 1:
            finalize()
