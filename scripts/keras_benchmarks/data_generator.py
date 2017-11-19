""" Generates input and label data for training models. """

import numpy as np


def generate_img_input_data(input_shape):
  """Generates training data and target labels.

  :param input_shape: input shape in the following format
                      (num_samples,channels, x, y)

  :return:
    numpy arrays: 'x_train, y_train'
  """
  x_train = np.random.randint(0, 255, input_shape)
  y_train = np.random.randint(0, 9, (input_shape[0],))

  return x_train, y_train

def generate_text_input_data(input_shape):
  """Generates training data and target labels.

  :param input_shape: input shape in the following format(num_samples, x, y)

  :return:
    numpy arrays: 'x_train, y_train' where the value of the arrays are booleans.
  """
  x_train = np.random.choice([True,False], input_shape)
  y_train = np.random.choice([True,False], (input_shape[0],input_shape[2]))

  return x_train, y_train
