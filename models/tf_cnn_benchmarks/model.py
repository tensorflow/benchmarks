"""Base model configuration for CNN benchmarks.
"""


class Model(object):
  def __init__(self, model, image_size, batch_size, layer_counts=None):
    self.model = model
    self.image_size = image_size
    self.batch_size = batch_size
    self.default_batch_size = batch_size
    self.layer_counts = layer_counts

  def get_model(self):
    return self.model

  def get_image_size(self):
    return self.image_size

  def get_batch_size(self):
    return self.batch_size

  def set_batch_size(self, batch_size):
    self.batch_size = batch_size

  def get_default_batch_size(self):
    return self.default_batch_size

  def get_layer_counts(self):
    return self.layer_counts

  def add_inference(self, cnn):
    raise ValueError('Must be implemented in derived classes')

