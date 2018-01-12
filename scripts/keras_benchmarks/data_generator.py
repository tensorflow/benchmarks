""" Generates input and label data for training models. """
import numpy as np


def generate_img_input_data(input_shape, num_classes):
  """Generates training data and target labels.

  # Arguments
    input_shape: input shape in the following format
                      `(num_samples, channels, x, y)`
    num_classes: number of classes that we want to classify the input

  # Returns
    numpy arrays: `x_train, y_train`
  """
  x_train = np.random.randint(0, 255, input_shape)
  y_train = np.random.randint(0, num_classes, (input_shape[0],))

  return x_train, y_train


def generate_text_input_data(input_shape, p=0.05, return_as_bool=True):
  """Generates training data and target labels .

  Given an input shape the function generates one hot encoded vectors. For
  example when we use words as our tokens, the presence/absence of the given
  word in the vocabulary is represented by True/False.

  # Arguments
    input_shape: input shape in the following format `(num_samples, x, y)`
    p: fraction of tokens that are present in the vocabulary
    return_as_bool: data and labels are returned as boolean arrays

  # Returns
    numpy arrays: `x_train, y_train`
  """
  x_train = np.random.binomial(1, p, input_shape)
  y_train = np.random.binomial(1, p, (input_shape[0], input_shape[2]))

  if return_as_bool:
    return x_train.astype(bool), y_train.astype(bool)

  return x_train, y_train

