"""Trivial model configuration."""

import model


class TrivialModel(model.Model):
  """Trivial model configuration."""

  def __init__(self):
    super(TrivialModel, self).__init__('trivial', 224 + 3, 32, 0.005)

  def add_inference(self, cnn):
    cnn.reshape([-1, 227 * 227 * 3])
    cnn.affine(1)
    cnn.affine(4096)
