'''
Model constructed using keras/applications.

Benchmark a Resnet50 model that implements subclasses.
'''

from __future__ import print_function

import keras

from models import timehistory
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from models import resnet50_layers

def crossentropy_from_logits(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(target=y_true,
                                                     output=y_pred,
                                                     from_logits=True)

def device_and_data_format():
    return ('/gpu:0', 'channels_first') if tfe.num_gpus() > 0 else ('/cpu:0',
                                                                    'channels_last')

class Resnet50SubclassBenchmark:

    def __init__(self):
        self.test_name = "resnet50_subclass"
        self.sample_type = "images"
        self.total_time = 0
        self.batch_size = 64
        self.epochs = 4
        self.num_samples = 1000
        self.test_type = 'tf.keras, subclass'

    def run_benchmark(self, gpus=0, use_dataset_tensors=False):
        if tf.keras.backend.backend() != "tensorflow" or use_dataset_tensors:
            print("You cannot run this model without the tensorflow backend enabled or with dataset tensors. ")
            return

        print("Running model ", self.test_name)
        tf.keras.backend.set_learning_phase(True)

        input_shape = (self.num_samples, 3, 224, 224)
        num_classes = 1000

        x_train = np.random.randint(0, 255, input_shape)
        y_train = np.random.randint(0, num_classes, (input_shape[0],))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        if gpus >= 1:
            tf.keras.backend.set_image_data_format('channels_first')
            shape = (3, 224, 224)

        if tf.keras.backend.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            shape = (224, 224, 3)

        print("data format is ", keras.backend.image_data_format())
        x_train /= 255
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')


        device, data_format = device_and_data_format()
        with tf.device(device):
            inputs = tf.keras.layers.Input(shape=shape)
            resnet50_model = resnet50_layers.ResNet50(data_format)
            outputs = resnet50_model(inputs, training=True)
            model = tf.keras.models.Model(inputs, outputs)
            model.compile(loss=crossentropy_from_logits,
                          optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-6),
                          metrics=['accuracy'])

            time_callback = timehistory.TimeHistory()

            model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                      shuffle=True, callbacks=[time_callback])

            self.total_time = 0
            for i in range(1, self.epochs):
                self.total_time += time_callback.times[i]

        tf.keras.backend.clear_session()


