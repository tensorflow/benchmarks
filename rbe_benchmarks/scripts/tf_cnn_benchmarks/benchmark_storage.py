"""Provides ways to store benchmark output."""


def store_benchmark(data, storage_type=None):
  """Store benchmark data.

  Args:
    data: Dictionary mapping from string benchmark name to
      numeric benchmark value.
    storage_type: (string) Specifies where to store benchmark
      result. If storage_type is
      'cbuild_benchmark_datastore': store outputs in our continuous
        build datastore. gcloud must be setup in current environment
        pointing to the project where data will be added.
  """
  if storage_type == 'cbuild_benchmark_datastore':
    try:
      # pylint: disable=g-import-not-at-top
      import cbuild_benchmark_storage
      # pylint: enable=g-import-not-at-top
    except ImportError:
      raise ImportError(
          'Missing cbuild_benchmark_storage.py required for '
          'benchmark_cloud_datastore option')
    cbuild_benchmark_storage.upload_to_benchmark_datastore(data)
  else:
    assert False, 'unknown storage_type: ' + storage_type
