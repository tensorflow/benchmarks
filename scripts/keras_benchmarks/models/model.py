from interface import Interface


class BenchmarkModelInterface(Interface):

  def get_totaltime(self):
    pass

  def get_iters(self):
    pass

  def get_testname(self):
    pass

  def get_sampletype(self):
    pass
