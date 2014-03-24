import os.path
import numpy as np
from node import *

import pycuda.gpuarray as gpuarray
from pyrap.tables import table

from RimeJonesBK import *
from RimeJonesEBK import *
from RimeJonesMultiply import *
from RimeJonesReduce import *

from pipedrimes import PipedRimes

import pycuda.autoinit 



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

		self.lm_shape = (2, nsrc)
		self.brightness_shape = (5, nsrc)
		self.point_errors_shape = (2, na, ntime)

		# Copy the uvw and wavelength data to the gpu
		self.uvw_gpu = gpuarray.to_gpu(self.uvw)
		self.wavelength_gpu = gpuarray.to_gpu(self.wavelength)

		# Allocate empty gpu arrays for lm, brightness and point errors
		self.lm_gpu = gpuarray.empty(self.lm_shape, dtype=ft)
		self.brightness_gpu = gpuarray.empty(self.brightness_shape, dtype=ft)
		self.point_errors_gpu = gpuarray.empty(self.point_errors_shape, \
			dtype=ft)

		# Output jones matrix
		self.jones_shape = (4,nbl,nchan,ntime,nsrc)
		self.jones_gpu = gpuarray.empty(self.jones_shape,dtype=ct)

		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		self.keys = (np.arange(np.product(self.jones_shape[:-1]))
			*self.jones_shape[-1]).astype(np.int32)
		self.keys_gpu = gpuarray.to_gpu(self.keys)

		# Output sum matrix
		self.sums_gpu = gpuarray.empty(self.keys.shape, dtype=ct)

		t.close()
		ta.close()
		tf.close()

	def set_point_errors(self, point_errors):
		sd = self
		if point_errors.shape != sd.point_errors_gpu.shape:
			raise ValueError, 'point_errors shape is wrong. Should be %s, but is %s.' % (sd.point_errors_gpu.shape, point_errors.shape,)
		
		#sd.point_errors = point_errors[:].astype(sd.ft)
		sd.point_errors_gpu.set(point_errors)

	def set_lm(self, lm):
		sd = self
		if lm.shape != sd.lm_gpu.shape:
			raise ValueError, 'lm shape is wrong. Should be %s, but is %s.' % (sd.lm_gpu.shape, lm.shape,)

		#sd.lm = lm[:].astype(sd.ft)
		sd.lm_gpu.set(lm)

	def set_brightness(self, brightness):
		sd = self
		if brightness.shape != sd.brightness_gpu.shape:
			raise ValueError, 'brightness shape is wrong. Should be %s' % (sd.brightness_gpu.shape, brightness.shape,)

		#sd.brightness = brightness[:].astype(sd.ft)
		sd.brightness_gpu.set(brightness)

	def get_jones(self):
		return self.sums_gpu.get()

	def configure(self):
		pass

if __name__ == '__main__':
	# Create a shared data object from the Measurement Set file
	sd = MeasurementSetSharedData('/home/simon/data/WSRT.MS', nsrc=100)
	# Create a pipeline consisting of an EBK kernel, followed by a reduction
	pipeline = PipedRimes([RimeJonesEBK(), RimeJonesReduce()])

	# Random point source coordinates in the l,m,n (brightness image) domain
	l=sd.ft(np.random.random(sd.nsrc)*0.1)
	m=sd.ft(np.random.random(sd.nsrc)*0.1)
	lm=np.array([l,m], dtype=sd.ft)

	# Random brightness matrix for the point sources
	fI=sd.ft(np.ones((sd.nsrc,)))
	fQ=sd.ft(np.random.random(sd.nsrc)*0.5)
	fU=sd.ft(np.random.random(sd.nsrc)*0.5)
	fV=sd.ft(np.random.random(sd.nsrc)*0.5)
	alpha=sd.ft(np.random.random(sd.nsrc)*0.1)
	brightness = np.array([fI,fQ,fV,fU,alpha], dtype=sd.ft)

	# Generate random antenna pointing errors
	point_errors = np.random.random(2*sd.na*sd.ntime)\
		.astype(sd.ft).reshape((2, sd.na, sd.ntime))

	# Set data on the shared data object. Uploads to GPU
	sd.set_lm(lm)
	sd.set_brightness(brightness)
	sd.set_point_errors(point_errors)

	kernels_start, kernels_end = cuda.Event(), cuda.Event()

	# Execute the pipeline
	for i in range(10):
		# Change parameters for this run
		sd.set_lm(lm)
		# Execute the pipeline
		kernels_start.record()
		pipeline.execute(sd)
		kernels_end.record()
		kernels_end.synchronize()
		# Obtain the Jones matrices (slow)

		print 'kernels: elapsed time: %gs' %\
			(kernels_start.time_till(kernels_end)*1e-3)

		jones = sd.get_jones()