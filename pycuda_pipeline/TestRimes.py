import unittest
from pipedrimes import RimeShared
from RimeJonesBK import RimeJonesBK
from RimeJonesReduce import RimeJonesReduce
from RimeJonesMultiplyInbuilt import RimeJonesMultiply
import numpy as np

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import predict
import segreduce


class TestRimes(unittest.TestCase):
	def setUp(self):
		np.random.seed(100)

		sd = self.shared_data = RimeShared()
		sd.configure()

		self.rime_bk = RimeJonesBK()
		self.rime_multiply = RimeJonesMultiply()
		self.rime_reduce = RimeJonesReduce()

		self.rime_bk.initialise(sd)
		self.rime_multiply.initialise(sd)
		self.rime_reduce.initialise(sd)

	def tearDown(self):
		sd = self.shared_data

		self.rime_bk.shutdown(sd)
		self.rime_multiply.shutdown(sd)
		self.rime_reduce.shutdown(sd)

		del self.rime_bk
		del self.rime_multiply
		del self.rime_reduce

	def test_BK(self):
		sd, rime_bk = self.shared_data, self.rime_bk

		baselines_per_block = 8 if sd.nbl > 8 else sd.nbl
		srcs_per_block = 64 if sd.nsrc > 64 else sd.nsrc

		baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
		src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block
		time_chan_blocks = sd.ntime*sd.nchan

		block=(baselines_per_block,srcs_per_block,1)
		grid=(baseline_blocks,src_blocks,time_chan_blocks)

		#print 'block', block, 'grid', grid

		rime_bk.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
		    sd.wavelength_gpu,  sd.jones_gpu,
		    np.int32(sd.nsrc), np.int32(sd.nbl),
		    np.int32(sd.nchan), np.int32(sd.ntime),
		    block=block, grid=grid,
		    shared=3*(baselines_per_block+srcs_per_block)*\
		    	np.dtype(np.float64).itemsize)

		# Repeat the wavelengths along the timesteps for now
		# dim nchan x ntime. This is a 1D array for now
		# as it makes broadcasting easier below. We reshape
		# it into nchan x ntime just before the final comparison
		w = np.repeat(sd.wavelength,sd.ntime)

		# n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
		n = np.sqrt(1. - sd.lma[0]**2 - sd.lma[1]**2) - 1.

		# u*l+v*m+w*n. Outer product creates array of dim nbl x nsrcs
		phase = np.outer(sd.uvw[0], sd.lma[0]) + \
			np.outer(sd.uvw[1], sd.lma[1]) + \
			np.outer(sd.uvw[2],n)

		# 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
		phase = (2*np.pi*1j*phase)[:,np.newaxis,:]/w[:,np.newaxis]
		# Dim nchan x ntime x nsrcs 
		power = np.power(1e6/w[:,np.newaxis], sd.lma[2])
		# This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
		phase_term = power*np.exp(phase)

		# Create the brightness matrix. Dim 4 x nsrcs
		sky = np.complex128([
			sd.sky[0]+sd.sky[3] + 0j,		# fI+fQ + 0j
			sd.sky[1] + 1j*sd.sky[2],		# fU + fV*1j
			sd.sky[1] - 1j*sd.sky[2],		# fU - fV*1j
			sd.sky[0]-sd.sky[3] + 0j])		# fI-fQ + 0j

		# This works due to broadcast! Multiplies along
		# srcs axis of sky. Dim 4 x nbl x nsrcs x nchan x ntime.
		# Also reshape the combined nchan and ntime axis into
		# two separate axes
		jones_cpu = (phase_term*sky[:,np.newaxis, np.newaxis,:])\
			.reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))

		# Get the jones matrices calculated by the GPU
		jones = sd.jones_gpu.get()

		# Test that the jones CPU calculation matches that of the GPU calculation
		self.assertTrue(np.allclose(jones_cpu, jones))

	@unittest.skipIf(False, 'test_multiply numpy code is somewhat inefficient')
	def test_multiply(self):
		sd, rime_multiply = self.shared_data, self.rime_multiply

		na=sd.na          # Number of antenna
		nbl=sd.nbl        # Number of baselines
		nchan=sd.nchan    # Number of channels
		nsrc=sd.nsrc      # Number of sources
		ntime=sd.ntime    # Number of timesteps

		# Output jones matrix
		njones = nbl*nsrc*nchan*ntime
		jsize = np.product(sd.jones_shape) # Number of complex  numbers
		# Create some random jones matrices to multiply together
		jones_lhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(np.complex128).reshape(sd.jones_shape)
		jones_rhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(np.complex128).reshape(sd.jones_shape)

		jones_per_block = 256 if njones > 256 else njones
		jones_blocks = (njones + jones_per_block - 1) / jones_per_block
		block, grid = (jones_per_block,1,1), (jones_blocks,1,1)

		#print 'block', block, 'grid', grid

		jones_lhs_gpu = gpuarray.to_gpu(jones_lhs)
		jones_rhs_gpu = gpuarray.to_gpu(jones_rhs)
		jones_output_gpu = gpuarray.empty(shape=sd.jones_shape, dtype=np.complex128)

		rime_multiply.kernel(jones_lhs_gpu, jones_rhs_gpu, jones_output_gpu,
			np.int32(njones), block=block, grid=grid,
		    shared=1*jones_per_block*np.dtype(np.complex128).itemsize)

		# Get the result off the gpu
		jones_output = jones_output_gpu.get()

		# Perform the calculation on the CPU
		jones_output_cpu = np.empty(shape=sd.jones_shape, dtype=np.complex128)

		# TODO: There must be a more numpy way to do this
		# Its dog slow...
		# Possible alternative to use with np.rollaxis:
		# from numpy.core.umath_tests import matrix_multiply
		# Doesn't work with complex numbers tho
		for bl in range(nbl):
			for ch in range(nchan):
				for t in range(ntime):	    		
					for src in range(nsrc):
						jones_output_cpu[:,bl,ch,t,src] = np.dot(
						jones_lhs[:,bl,ch,t,src].reshape(2,2),
						jones_rhs[:,bl,ch,t,src].reshape(2,2)).reshape(4)

		# Confirm similar results
		self.assertTrue(np.allclose(jones_output, jones_output_cpu))

	def test_reduce(self):
		sd, rime_reduce = self.shared_data, self.rime_reduce

		# Create the jones matrices
		jsize = np.product(sd.jones_shape)
		jones = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(np.complex128).reshape(sd.jones_shape)

		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		keys = (np.arange(np.product(sd.jones_shape[:-1]))*sd.jones_shape[-1])\
			.astype(np.int32)
		
		# Send the jones and keys to the gpu, and create the output array for
		# the segmented sums
		jones_gpu = gpuarray.to_gpu(jones.flatten())
		keys_gpu = gpuarray.to_gpu(keys)
		sums_gpu = gpuarray.empty(shape=keys.shape, dtype=jones.dtype.type)

		segreduce.segmented_reduce_complex128_sum(
			data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
			device_id=0)

		# Add everything along the last axis (time)
		sums_cpu = np.sum(jones,axis=len(sd.jones_shape)-1)

		# Confirm similar results
		self.assertTrue(np.allclose(sums_cpu.flatten(), sums_gpu.get()))

	def test_predict(self):
		sd, rime_bk, rime_reduce = self.shared_data, self.rime_bk, self.rime_reduce

		na=sd.na          # Number of antenna
		nbl=sd.nbl        # Number of baselines
		nchan=sd.nchan    # Number of channels
		nsrc=sd.nsrc      # Number of sources
		ntime=1 		  # Number of timesteps

		jones_shape=(4,nbl,nchan,ntime,nsrc)

		# Visibilities ! has to have double complex
		Vis=np.complex128(np.zeros((nbl,nchan,4)))
		# UVW coordinates
		uvw=sd.uvw.T.copy()

		# Frequencies in Hz
		WaveL = sd.wavelength
		# Sky coordinates
		lms=np.array([sd.lma[0], sd.lma[1], sd.sky[0], sd.lma[2], 
			sd.sky[3], sd.sky[1], sd.sky[2]]).astype(np.float64).T.copy()

		# Antennas
		A0=np.int64(np.random.rand(nbl)*na)
		A1=np.int64(np.random.rand(nbl)*na)

		# Create a the jones matrices, but make them identity matrices
		Sols=np.ones((nsrc,nchan,na))[:,:,:,np.newaxis]*(np.eye(2).reshape(4)).astype(np.complex128)

		# Matrix containing information, here just the reference frequency
		# to estimate the flux from spectral index
		Info=np.array([1e6],np.float64)

		# Call Cyrils' predict code
		P1=predict.predictSolsPol(Vis, A0, A1, uvw, lms, WaveL, Sols, Info)

		# Call the GPU RimeJonesBK node. First set up the grid parameters
		baselines_per_block = 8 if nbl > 8 else nbl
		srcs_per_block = 64 if nsrc > 64 else nsrc

		baseline_blocks = (nbl + baselines_per_block - 1) / baselines_per_block
		src_blocks = (nsrc + srcs_per_block - 1) / srcs_per_block
		time_chan_blocks = sd.ntime*sd.nchan

		block=(baselines_per_block,srcs_per_block,1)
		grid=(baseline_blocks,src_blocks,time_chan_blocks)

		jones_gpu = gpuarray.empty(jones_shape,dtype=np.complex128)

		# Invoke the kernel
		rime_bk.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
		    sd.wavelength_gpu,  jones_gpu,
		    np.int32(nsrc), np.int32(nbl),
		    np.int32(nchan), np.int32(ntime),
		    block=block, grid=grid,
		    shared=3*(baselines_per_block+srcs_per_block)*\
		    	np.dtype(np.float64).itemsize)

		# Set up the segmented reduction
		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		keys = (np.arange(np.product(jones_shape[:-1]))*jones_shape[-1])\
			.astype(np.int32).reshape(jones_shape[:-2])
		
		# Send the keys to the gpu, and create the output array for
		# the segmented sums
		keys_gpu = gpuarray.to_gpu(keys)
		sums_gpu = gpuarray.empty(shape=keys.shape, dtype=sd.jones_gpu.dtype.type)

		# Invoke the kernel
		segreduce.segmented_reduce_complex128_sum(
			data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
			device_id=0)

		# Shift the gpu jones matrices so they are on the last axis
		sums_cpu = np.rollaxis(sums_gpu.get(),0,len(jones_shape)-2)

		# Compare the GPU solution with Cyril's predict code
		self.assertTrue(np.allclose(sums_cpu, Vis))


if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(TestRimes)
	unittest.TextTestRunner(verbosity=2).run(suite)