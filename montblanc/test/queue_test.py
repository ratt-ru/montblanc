import threading
import pyrap.tables
import Queue

req_data_q = Queue.Queue()
rsp_data_q = Queue.Queue()

class MSDataRequest(object):
	def __init__(self, colname, startrow, nrow):
		self.colname = colname
		self.startrow = startrow
		self.nrow = nrow

def read_MS():
	t = pyrap.tables.table('/home/simon/data/WSRT.MS/')

	while True:
		item = req_data_q.get()
		if type(item) is int and item == -1:
			break
		data = t.getcol(item.colname,startrow=item.startrow,nrow=item.nrow)
		print 'Read %s' % (data.shape,)
		rsp_data_q.put(data)
		req_data_q.task_done()

	t.close()

	print 'MS reader thread shutting down'


def requester():
	for i in range(3):
		req_data_q.put(MSDataRequest('UVW',i*10,i*10+10))

	while True:
		try:
			item = rsp_data_q.get(timeout=1)
		except Queue.Empty:
			break
		print 'Got item'

		if type(item) is int and item == -1:
			break

		rsp_data_q.task_done()

	print 'Requester thread shutting down'

req_thread = threading.Thread(target=requester)
req_thread.start()

ms_read_thread = threading.Thread(target=read_MS)
ms_read_thread.start()

req_data_q.put(-1)

req_thread.join()
ms_read_thread.join()

print 'Exiting'