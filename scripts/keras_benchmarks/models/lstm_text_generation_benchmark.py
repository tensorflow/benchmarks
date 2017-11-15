'''
Original Model from keras/examples

Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

import time
from model import BenchmarkModel
from models import timehistory
if keras.backend.backend() == 'cntk':
  from gpu_mode import cntk_gpu_mode_config
from data_generator import generate_text_input_data

class LstmTextGenBenchmark(BenchmarkModel):

    def __init__(self):
        self._test_name = "lstm_text_generation"
        self._sample_type="text"
        self._total_time = 0
        self._batch_size = 128
        self._epochs = 2
        self._num_samples = 1000

    def run_benchmark(self, gpus=0):
        maxlen = 40
        char_len = 60

        input_shape = (self._num_samples, maxlen, 60)
        x, y = generate_text_input_data(input_shape)

        # build the model: a single LSTM
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, char_len)))
        model.add(Dense(char_len))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # create a distributed trainer for cntk
        if keras.backend.backend() is "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model, x.shape[0])
            x = x[start: end]
            y = y[start: end]

        # train the model, output generated text after each iteration
        start_time = time.time()
        time_callback = timehistory.TimeHistory()

        model.fit(x, y,
                    batch_size=self._batch_size,
                    epochs=self._epochs,
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