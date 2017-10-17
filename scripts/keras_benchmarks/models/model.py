
class BenchmarkModel(object):

  def get_totaltime(self):
    raise ValueError('Must be implemented in derived classes')

  def get_iters(self):
    raise ValueError('Must be implemented in derived classes')

  def get_testname(self):
    raise ValueError('Must be implemented in derived classes')

  def get_sampletype(self):
    raise ValueError('Must be implemented in derived classes')
