import os.path
import numpy as np
from BaseSharedData import *

import pycuda.gpuarray as gpuarray
from pyrap.tables import table

from RimeEBKFloat import *
from RimeReduce import *

from pipedrimes import PipedRimes

import pycuda.autoinit 

class MeasurementSetSharedData(GPUSharedData):
	ANTENNA_TABLE = "ANTENNA"
	SPECTRAL_WINDOW = "SPECTRAL_WINDOW"

	def __init__(self, ms_file, nsrc, dtype=np.float64):
		# Do some checks on the supplied filename
		if not isinstance(ms_file, str):
			raise TypeError, 'ms_file is not a string'

		if not os.path.isdir(ms_file):
			raise ValueError, '%s does not appear to be a valid path' % ms_file

		# Store the measurement set filename
		self.ms_file = ms_file

		# Open the measurement set
		t = table(self.ms_file, ack=False)

		# Get the UVW coordinates
		self.uvw=t.getcol("UVW").T.astype(dtype)

		# Open the antenna table
		ta=table(self.ms_file + os.sep + MeasurementSetSharedData.ANTENNA_TABLE, ack=False)
		# Open the spectral window table
		tf=table(self.ms_file + os.sep + MeasurementSetSharedData.SPECTRAL_WINDOW, ack=False)
		f=tf.getcol("CHAN_FREQ").astype(dtype)

		na = len(ta.getcol("NAME"))
		nbl = (na**2+na)/2
		nchan = f.size
		ntime = self.uvw.shape[1] / nbl

		super(MeasurementSetSharedData, self).__init__(\
			na=na,nchan=nchan,ntime=ntime,nsrc=nsrc,dtype=dtype)

		self.wavelength = 3e8/f

		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		self.keys = (np.arange(np.product(self.jones_shape[:-1]))
			*self.jones_shape[-1]).astype(np.int32)
		self.keys_gpu = gpuarray.to_gpu(self.keys)

		t.close()
		ta.close()
		tf.close()

	def get_visibilities(self):
		return self.vis_gpu.get()

if __name__ == '__main__':
	# Create a shared data object from the Measurement Set file
	sd = MeasurementSetSharedData('/home/simon/data/WSRT.MS', nsrc=100, dtype=np.float32)
	# Create a pipeline consisting of an EBK kernel, followed by a reduction
	pipeline = PipedRimes([RimeEBKFloat(), RimeReduceFloat()])

	print 'Using', sd.gpu_mem()/(1024*1024), 'MB of GPU memory.'

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
	brightness = np.array([fI,fQ,fU,fV,alpha], dtype=sd.ft)

	# Generate random antenna pointing errors
	point_errors = np.random.random(2*sd.na*sd.ntime)\
		.astype(sd.ft).reshape((2, sd.na, sd.ntime))

	# Set data on the shared data object. Uploads to GPU
	sd.transfer_lm(lm)
	sd.transfer_brightness(brightness)
	sd.transfer_point_errors(point_errors)

	kernels_start, kernels_end = cuda.Event(), cuda.Event()

	# Execute the pipeline
	for i in range(10):
		# Change parameters for this run
		sd.transfer_lm(lm)
		# Execute the pipeline
		kernels_start.record()
		pipeline.execute(sd)
		kernels_end.record()
		kernels_end.synchronize()
		# Obtain the Jones matrices (slow)

		print 'kernels: elapsed time: %gs' %\
			(kernels_start.time_till(kernels_end)*1e-3)

		V = sd.get_visibilities()