import os.path
import numpy as np
from node import *
from pyrap.tables import table

class MeasurementSetSharedData(SharedData):
	ANTENNA_TABLE = "ANTENNA"
	SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

	na = Parameter(7)
	nbl = Parameter((7*6)/2)
	nchan = Parameter(32)
	nsrc = Parameter(200)
	ntime = Parameter(10)

	def __init__(self, ms_filename, float_dtype=np.float64):
		super(MeasurementSetSharedData, self).__init__()

		# Do some checks on the supplied filename
		if not isinstance(ms_filename, str):
			raise TypeError, 'ms_filename is not a string'

		if not os.path.isdir(ms_filename):
			raise ValueError, '%s does not appear to be a valid path' % ms_filename

		# Store it
		self.ms_filename = ms_filename

		# Do some checks on the supplied float type
		# Set up the complex type
		if float_dtype == np.float32:
			self.ct = np.complex64
		elif float_dtype == np.float64:
			self.ct = np.complex128
		else:
			raise TypeError, 'Must specify either np.float32 or np.float64 for float_dtype'

		# Store it
		self.ft = float_dtype

		# Open the measurement set
		t = table(self.ms_filename, ack=False)

		# Get the UVW coordinates
		uvw=t.getcol("UVW").T.astype(self.ft)

		# Open the antenna table
		ta=table(self.ms_filename + os.sep + MeasurementSetSharedData.ANTENNA_TABLE, ack=False)
		# Open the spectral window table
		tf=table(self.ms_filename + os.sep + MeasurementSetSharedData.SPECTRAL_WINDOW, ack=False)
		f=tf.getcol("CHAN_FREQ").astype(self.ft)

		na = len(ta.getcol("NAME"))
		nbl = (na**2+na)/2
		nchan = f.size
		ntime = uvw.shape[1] / nbl
		self.wavelength = 3e8/self.ft(f)

		print "Number of Antenna", na
		print "Number of Baselines", nbl
		print "Number of Channels", nchan
		print "Number of Timesteps", ntime
		print "UVW shape" , uvw.shape, uvw.size

		t.close()
		ta.close()
		tf.close()

	def configure(self):
		pass

if __name__ == '__main__':
	sd = MeasurementSetSharedData('/home/simon/data/WSRT.MS')
	sd.configure()