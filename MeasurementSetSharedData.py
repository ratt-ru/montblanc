import os.path
import numpy as np
from node import *

import pycuda.gpuarray as gpuarray
from pyrap.tables import table

class MeasurementSetSharedData(SharedData):
	ANTENNA_TABLE = "ANTENNA"
	SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

	na = Parameter(7)
	nbl = Parameter((7*6)/2)
	nchan = Parameter(32)
	nsrc = Parameter(200)
	ntime = Parameter(10)

	def __init__(self, ms_file, nsrc, float_dtype=np.float64):
		super(MeasurementSetSharedData, self).__init__()

		# Do some checks on the supplied filename
		if not isinstance(ms_file, str):
			raise TypeError, 'ms_file is not a string'

		if not os.path.isdir(ms_file):
			raise ValueError, '%s does not appear to be a valid path' % ms_file

		# Store it
		self.ms_file = ms_file

		# Do some checks on the supplied float type
		# Set up the complex type
		if float_dtype == np.float32:
			self.ct = ct = np.complex64
		elif float_dtype == np.float64:
			self.ct = ct = np.complex128
		else:
			raise TypeError, 'Must specify either np.float32 or np.float64 for float_dtype'

		# Store it
		self.ft = ft = float_dtype

		# Open the measurement set
		t = table(self.ms_file, ack=False)

		# Get the UVW coordinates
		self.uvw=t.getcol("UVW").T.astype(self.ft)

		# Open the antenna table
		ta=table(self.ms_file + os.sep + MeasurementSetSharedData.ANTENNA_TABLE, ack=False)
		# Open the spectral window table
		tf=table(self.ms_file + os.sep + MeasurementSetSharedData.SPECTRAL_WINDOW, ack=False)
		f=tf.getcol("CHAN_FREQ").astype(self.ft)

		na = len(ta.getcol("NAME"))
		nbl = (na**2+na)/2
		nchan = f.size
		ntime = self.uvw.shape[1] / nbl
		self.wavelength = 3e8/ft(f)

		print "Number of Antenna", na
		print "Number of Baselines", nbl
		print "Number of Channels", nchan
		print "Number of Timesteps", ntime
		print "UVW shape" , self.uvw.shape, self.uvw.size

		self.na = na
		self.nbl = nbl
		self.nchan = nchan
		self.ntime = ntime
		self.nsrc = nsrc

		self.lma_shape = (3, nsrc)
		self.sky_shape = (4, nsrc)
		self.point_errors_shape = (2, na, ntime)

		# Copy the uvw, lma and sky data to the gpu
		self.uvw_gpu = gpuarray.to_gpu(self.uvw)
		self.wavelength_gpu = gpuarray.to_gpu(self.wavelength)

		# Allocate empty gpu arrays for lma, sky and point errors
		self.lma_gpu = gpuarray.empty(self.lma_shape, dtype=ft)
		self.sky_gpu = gpuarray.empty(self.sky_shape, dtype=ft)
		self.point_errors_gpu = gpuarray.empty(self.point_errors_shape, \
			dtype=ft)

		# Output jones matrix
		self.jones_shape = (4,nbl,nchan,ntime,nsrc)
		self.jones_gpu = gpuarray.empty(self.jones_shape,dtype=ct)

		t.close()
		ta.close()
		tf.close()

	def set_point_errors(self, point_errors):
		sd = self
		if point_errors.shape != sd.point_errors_gpu.shape:
			raise ValueError, 'point_errors shape is wrong. Should be %s, but is %s.' % (sd.point_errors_gpu.shape, point_errors.shape,)
		
		#sd.point_errors = point_errors[:].astype(sd.ft)
		sd.point_errors_gpu.set(point_errors)

	def set_lma(self, lma):
		sd = self
		if lma.shape != sd.lma_gpu.shape:
			raise ValueError, 'lma shape is wrong. Should be %s, but is %s.' % (sd.lma_gpu.shape, lma.shape,)

		#sd.lma = lma[:].astype(sd.ft)
		sd.lma_gpu.set(lma)

	def set_sky(self, sky):
		sd = self
		if sky.shape != sd.sky_gpu.shape:
			raise ValueError, 'sky shape is wrong. Should be %s' % (sd.sky_gpu.shape, sky.shape,)

		#sd.sky = sky[:].astype(sd.ft)
		sd.sky_gpu.set(sky)

	def configure(self):
		pass

if __name__ == '__main__':
	import pycuda.autoinit 

	sd = MeasurementSetSharedData('/home/simon/data/WSRT.MS', nsrc=100)

	# Point source coordinates in the l,m,n (sky image) domain
	l=sd.ft(np.random.random(sd.nsrc)*0.1)
	m=sd.ft(np.random.random(sd.nsrc)*0.1)
	alpha=sd.ft(np.random.random(sd.nsrc)*0.1)
	lma=np.array([l,m,alpha], dtype=sd.ft)

	# Brightness matrix for the point sources
	fI=sd.ft(np.ones((sd.nsrc,)))
	fV=sd.ft(np.random.random(sd.nsrc)*0.5)
	fU=sd.ft(np.random.random(sd.nsrc)*0.5)
	fQ=sd.ft(np.random.random(sd.nsrc)*0.5)
	sky = np.array([fI,fV,fU,fQ], dtype=sd.ft)

	# Generate the antenna pointing errors
	point_errors = np.random.random(2*sd.na*sd.ntime)\
		.astype(sd.ft).reshape((2, sd.na, sd.ntime))

	sd.set_lma(lma)
	sd.set_sky(sky)
	sd.set_point_errors(point_errors)

	sd.configure()