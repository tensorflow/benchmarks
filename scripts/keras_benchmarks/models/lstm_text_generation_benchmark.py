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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np

import time
from model import BenchmarkModel
from models import timehistory


class LstmTextGenBenchmark(BenchmarkModel):

  # TODO(anjalisridhar): you can pass test name and sample type when creating
  # the object
  def __init__(self):
    self._test_name = "lstm_text_generation"
    self._sample_type="text"
    self._total_time = 0
    self._batch_size = 128
    self._epochs = 2

  def benchmarkLstmTextGen(self, keras_backend=None, gpu_count=0):
    if keras_backend is None:
      raise ValueError('keras_backend parameter must be specified.')

    if keras_backend is not "tensorflow" and gpu_count > 0:
      raise ValueError('gpu mode is currently only supported for tensorflow backends.')

    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
      sentences.append(text[i: i + maxlen])
      next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
      for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
      y[i, char_indices[next_chars[i]]] = 1


    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # train the model, output generated text after each iteration
    start_time = time.time()
    time_callback = timehistory.TimeHistory()

    for iteration in range(1, 60):
      print()
      print('-' * 50)
      print('Iteration', iteration)

      model.fit(x, y,
                batch_size=self._batch_size,
                epochs=self._epochs)

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