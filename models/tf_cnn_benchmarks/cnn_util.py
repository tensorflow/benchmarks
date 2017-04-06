"""Utilities for CNN benchmarks.
"""

import tensorflow as tf


def tensorflow_version_tuple():
  # TODO: change this to not be hardcoded.
  # v = tf.__version__
  v = '10.1.2'
  major, minor, patch = v.split('.')
  return (int(major), int(minor), patch)


def tensorflow_version():
  vt = tensorflow_version_tuple()
  return vt[0] * 1000 + vt[1]

