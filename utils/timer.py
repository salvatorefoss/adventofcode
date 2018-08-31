import time


class Timer(object):
  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *args):
    self.end = time.time()
    self.secs = self.end - self.start
    print('{0:06f} ms'.format(1000*self.secs))