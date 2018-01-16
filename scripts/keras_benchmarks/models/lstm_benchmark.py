'''
Original Model from keras/examples/lstm_text_generation.py

Benchmark for a LSTM model.
'''
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from models import timehistory
from data_generator import generate_text_input_data

if keras.backend.backend() == 'cntk':
  from gpu_mode import cntk_gpu_mode_config


class LstmBenchmark():

    def __init__(self):
        self.test_name = "lstm"
        self.sample_type = "text"
        self.total_time = 0
        self.batch_size = 128
        self.epochs = 2
        self.num_samples = 1000

    def run_benchmark(self, gpus=0):
        input_dim_1 = 40
        input_dim_2 = 60

        input_shape = (self.num_samples, input_dim_1, 60)
        x, y = generate_text_input_data(input_shape)

        # build the model: a single LSTM
        model = Sequential()
        model.add(LSTM(128, input_shape=(input_dim_1, input_dim_2)))
        model.add(Dense(input_dim_2), activation='softmax')

        optimizer = RMSprop(lr=0.01)

        if keras.backend.backend() is "tensorflow" and gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # create a distributed trainer for cntk
        if keras.backend.backend() is "cntk" and gpus > 1:
            start, end = cntk_gpu_mode_config(model, x.shape[0])
            x = x[start: end]
            y = y[start: end]

        time_callback = timehistory.TimeHistory()

        model.fit(x, y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[time_callback])

        self.total_time = 0
        for i in range(1, self.epochs):
            self.total_time += time_callback.times[i]
