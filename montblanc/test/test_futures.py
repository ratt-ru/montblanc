import pyrap.tables
import numpy as np
import concurrent.futures as cf
import time

class timer(object):
    def __init__(self):
      self.reset()
    def __enter__(self):
      self.start()
      return self
    def __exit__(self, *args):
      self.stop()
    def start(self):
      self._start = time.time()
    def stop(self):
      end = time.time()
      self._secs += end - self._start
    def elapsed(self):
      return self._secs
    def reset(self):
      self._secs = 0

t = pyrap.tables.table('/home/simon/data/WSRT.MS')
nrows = t.nrows()
size = 2000

c=0

def inc():
	global c
	c += 1
	return c

OT = timer()

OT.start()

nworkers = 8
njobs = 100

with cf.ThreadPoolExecutor(1) as io, \
	cf.ThreadPoolExecutor(max_workers=nworkers) as req, \
	cf.ThreadPoolExecutor(1) as counter:
	
	def do_requests():
		T = timer()
		T.start()
		for i in range(10):
			startrow=np.random.randint(0,nrows-size)
			future1 = io.submit(t.getcol,'UVW',startrow=startrow,nrow=size)
			future2 = io.submit(t.getcoldesc,'UVW')
			uvw = future1.result()
			a = np.random.random(1024*1024)*5
			a = np.rollaxis(a.reshape(1024,1024),start=1,axis=0)
			b = np.exp(a).sum()
			b *= np.exp(a+1).sum()			
			b *= np.exp(a+2).sum()			

			#print '+',
			counter.submit(inc)
		T.stop()
		print 'T S: %d E: %f' % (T._start, T.elapsed())
		T.reset()

	futures = []

	for i in range(njobs):
		futures.append(req.submit(do_requests))

	# Need for all futures to return correctly!
	for f in futures: f.result()

io.shutdown(wait=True)
req.shutdown(wait=True)
counter.shutdown(wait=True)

OT.stop()
print 'OT S: %d E: %f' % (OT._start, OT.elapsed())

print c