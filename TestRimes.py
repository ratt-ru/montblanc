import logging
import unittest
import numpy as np
import time
import sys

from RimeJonesBK import RimeJonesBK
from RimeJonesBKFloat import RimeJonesBKFloat
from RimeJonesEBK import RimeJonesEBK
from RimeJonesEBKFloat import RimeJonesEBKFloat
from RimeJonesReduce import RimeJonesReduce
from RimeJonesMultiply import RimeJonesMultiply
from RimeChiSquaredFloat import RimeChiSquaredFloat
from TestRimeSharedData import TestRimeSharedData

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import predict
import crimes


class TestRimes(unittest.TestCase):
	def setUp(self):
		np.random.seed(100)
		# Set up various things that aren't possible in PyCUDA
		crimes.setup_cuda()

	def tearDown(self):
		pass

	def test_BK_float(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=10,nsrc=200,dtype=np.float32)		
		rime_bk = RimeJonesBKFloat()

		rime_bk.initialise(sd)

		rime_bk.execute(sd)

		# Shutdown the rime_bk node, we don't need it any more
		rime_bk.shutdown(sd)

		# Repeat the wavelengths along the timesteps for now
		# dim nchan x ntime. 
		w = np.repeat(sd.wavelength,sd.ntime).reshape(sd.nchan, sd.ntime)

		# n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
		n = np.sqrt(1. - sd.lm[0]**2 - sd.lm[1]**2) - 1.

		# u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
		phase = (np.outer(sd.uvw[0], sd.lm[0]) + \
			np.outer(sd.uvw[1], sd.lm[1]) + \
			np.outer(sd.uvw[2],n))\
				.reshape(sd.nbl, sd.ntime, sd.nsrc)

		# 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
		phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
		# Dim nchan x ntime x nsrcs 
		power = np.power(1e6/w[:,:,np.newaxis], sd.brightness[4])
		# This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
		phase_term = power*np.exp(phase)

		# Create the brightness matrix. Dim 4 x nsrcs
		brightness = sd.ct([
			sd.brightness[0]+sd.brightness[1] + 0j,		# fI+fQ + 0j
			sd.brightness[2] + 1j*sd.brightness[3],		# fU + fV*1j
			sd.brightness[2] - 1j*sd.brightness[3],		# fU - fV*1j
			sd.brightness[0]-sd.brightness[1] + 0j])		# fI-fQ + 0j

		# This works due to broadcast! Multiplies along
		# srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
		# Also reshape the combined nchan and ntime axis into
		# two separate axes
		jones_cpu = (phase_term[np.newaxis,:,:,:,:]* \
			brightness[:,np.newaxis, np.newaxis, np.newaxis,:])\
			.reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))

		# Get the jones matrices calculated by the GPU
		jones = sd.jones_gpu.get()

		# Test that the jones CPU calculation matches that of the GPU calculation
		self.assertTrue(np.allclose(jones_cpu, jones))

	def test_BK(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=10,nsrc=200)		
		rime_bk = RimeJonesBK()

		rime_bk.initialise(sd)

		rime_bk.execute(sd)

		# Shutdown the rime_bk node, we don't need it any more
		rime_bk.shutdown(sd)

		# Repeat the wavelengths along the timesteps for now
		# dim nchan x ntime. 
		w = np.repeat(sd.wavelength,sd.ntime).reshape(sd.nchan, sd.ntime)

		# n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
		n = np.sqrt(1. - sd.lm[0]**2 - sd.lm[1]**2) - 1.

		# u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
		phase = (np.outer(sd.uvw[0], sd.lm[0]) + \
			np.outer(sd.uvw[1], sd.lm[1]) + \
			np.outer(sd.uvw[2],n))\
				.reshape(sd.nbl, sd.ntime, sd.nsrc)

		# 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
		phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
		# Dim nchan x ntime x nsrcs 
		power = np.power(1e6/w[:,:,np.newaxis], sd.brightness[4])
		# This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
		phase_term = power*np.exp(phase)

		# Create the brightness matrix. Dim 4 x nsrcs
		brightness = sd.ct([
			sd.brightness[0]+sd.brightness[1] + 0j,		# fI+fQ + 0j
			sd.brightness[2] + 1j*sd.brightness[3],		# fU + fV*1j
			sd.brightness[2] - 1j*sd.brightness[3],		# fU - fV*1j
			sd.brightness[0]-sd.brightness[1] + 0j])		# fI-fQ + 0j

		# This works due to broadcast! Multiplies along
		# srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
		# Also reshape the combined nchan and ntime axis into
		# two separate axes
		jones_cpu = (phase_term[np.newaxis,:,:,:,:]* \
			brightness[:,np.newaxis, np.newaxis, np.newaxis,:])\
			.reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))

		# Get the jones matrices calculated by the GPU
		jones = sd.jones_gpu.get()

		# Test that the jones CPU calculation matches that of the GPU calculation
		self.assertTrue(np.allclose(jones_cpu, jones))

	def test_EBK(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=10,nsrc=200)		
		rime_ebk = RimeJonesEBK()
		rime_bk = RimeJonesBK()

		rime_ebk.initialise(sd)
		rime_bk.initialise(sd)

		# Invoke the BK kernel
		rime_bk.execute(sd)

		jones_cpu = sd.jones_gpu.get()

		rime_ebk.execute(sd)

		rime_ebk.shutdown(sd)
		rime_bk.shutdown(sd)

	def test_EBK_float(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=10,nsrc=200,dtype=np.float32)		
		rime_ebk = RimeJonesEBKFloat()
		rime_bk = RimeJonesBKFloat()

		rime_ebk.initialise(sd)
		rime_bk.initialise(sd)

		# Invoke the BK kernel
		rime_bk.execute(sd)

		jones_cpu = sd.jones_gpu.get()

		for bl in range(sd.nbl):
			ANT1 = int(np.floor((np.sqrt(1+8*bl)-1)/2))
			ANT2 = ANT1*(ANT1-1)/2
			ANT1 += 1;

		"""
		for bl in range(sd.nbl):
			ANT1 = int(np.floor((np.sqrt(1+8*bl)-1)/2))
			ANT2 = ANT1*(ANT1-1)/2
			ANT1 += 1;

			for chan in range(sd.nchan):
				wave = sd.wavelength[chan]
				for time in range(sd.ntime):
					for src in range(sd.nsrc):
						l, m = sd.lm[0][src]

						jones[4:,bl,chan,time,src]
		"""

		

		rime_ebk.execute(sd)		

		rime_ebk.shutdown(sd)
		rime_bk.shutdown(sd)

	@unittest.skipIf(False, 'test_multiply numpy code is somewhat inefficient')
	def test_multiply(self):
		# Make the problem size smaller, due to slow numpy code later on
		sd = TestRimeSharedData(na=5,nchan=4,ntime=2,nsrc=10)
		rime_multiply = RimeJonesMultiply()

		rime_multiply.initialise(sd)

		# Output jones matrix
		njones = sd.nbl*sd.nsrc*sd.nchan*sd.ntime
		jsize = np.product(sd.jones_shape) # Number of complex  numbers
		# Create some random jones matrices to multiply together
		jones_lhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(sd.ct).reshape(sd.jones_shape)
		jones_rhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(sd.ct).reshape(sd.jones_shape)

		jones_lhs_gpu = gpuarray.to_gpu(jones_lhs)
		jones_rhs_gpu = gpuarray.to_gpu(jones_rhs)

		rime_multiply.kernel(jones_lhs_gpu, jones_rhs_gpu, sd.jones_gpu,
			np.int32(njones), **rime_multiply.get_kernel_params(sd))

		# Shutdown the rime node, we don't need it any more
		rime_multiply.shutdown(sd)

		# Get the result off the gpu
		jones_output_gpu = sd.jones_gpu.get()

		# Perform the calculation on the CPU
		jones_output_cpu = np.empty(shape=sd.jones_shape, dtype=sd.ct)

		# TODO: There must be a more numpy way to do this
		# Its dog slow...
		# Possible alternative to use with np.rollaxis:
		# from numpy.core.umath_tests import matrix_multiply
		# Doesn't work with complex numbers tho
		for bl in range(sd.nbl):
			for ch in range(sd.nchan):
				for t in range(sd.ntime):	    		
					for src in range(sd.nsrc):
						jones_output_cpu[:,bl,ch,t,src] = np.dot(
						jones_lhs[:,bl,ch,t,src].reshape(2,2),
						jones_rhs[:,bl,ch,t,src].reshape(2,2)).reshape(4)

		# Confirm similar results
		self.assertTrue(np.allclose(jones_output_gpu, jones_output_cpu))

	def test_chi_squared(self):
		sd = TestRimeSharedData(na=100,nchan=64,ntime=20,nsrc=1,dtype=np.float32)

		rime_X_2 = RimeChiSquaredFloat()
		rime_X_2.initialise(sd)

		# Run the kernel
		rime_X_2.execute(sd)

		# Take the difference between the visibilities and the model
		# (4,nbl,nchan,ntime)
		d = sd.vis - sd.bayes_model
		# Reduces a dimension so that we have (nbl,nchan,ntime)
		# (XX.real^2 + XY.real^2 + YX.real^2 + YY.real^2) + ((XX.imag^2 + XY.imag^2 + YX.imag^2 + YY.imag^2))
		chi_sqrd_cpu = (np.add.reduce(d.real**2,axis=0) + np.add.reduce(d.imag**2,axis=0))/sd.sigma_sqrd
		chi_sqrd_gpu = sd.chi_sqrd_result_gpu.get()

		assert np.allclose(chi_sqrd_cpu, chi_sqrd_gpu)

		rime_X_2.shutdown(sd)

	def test_reduce(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=10,nsrc=200)		
		rime_reduce = RimeJonesReduce()

		rime_reduce.initialise(sd)

		# Create the jones matrices
		jsize = np.product(sd.jones_shape)
		jones = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(sd.ct).reshape(sd.jones_shape)

		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		keys = (np.arange(np.product(sd.jones_shape[:-1]))*sd.jones_shape[-1])\
			.astype(np.int32)
		
		# Send the jones and keys to the gpu, and create the output array for
		# the visibilities.
		jones_gpu = gpuarray.to_gpu(jones.flatten())
		keys_gpu = gpuarray.to_gpu(keys)
		vis_gpu = gpuarray.empty(shape=keys.shape, dtype=jones.dtype.type)

		crimes.segmented_reduce_complex128_sum(
			data=jones_gpu, seg_starts=keys_gpu, seg_sums=vis_gpu,
			device_id=0)

		# Shutdown the rime node, we don't need it any more
		rime_reduce.shutdown(sd)

		# Add everything along the last axis (time)
		vis_cpu = np.sum(jones,axis=len(sd.jones_shape)-1)

		# Confirm similar results
		self.assertTrue(np.allclose(vis_cpu.flatten(), vis_gpu.get()))

	def time_predict(self, sd, log):
		na=sd.na          # Number of antenna
		nbl=sd.nbl        # Number of baselines
		nchan=sd.nchan    # Number of channels
		nsrc=sd.nsrc      # Number of sources
		ntime=sd.ntime	  # Number of timesteps

		# Visibilities ! has to have double complex
		Vis=np.complex128(np.zeros((nbl,nchan,4)))
		# UVW coordinates
		uvw=sd.uvw.T.astype(np.float64).copy()

		# Frequencies in Hz
		WaveL = sd.wavelength.astype(np.float64)
		# Sky coordinates
		lms=np.array([sd.lm[0], sd.lm[1], sd.brightness[0], sd.brightness[4], 
			sd.brightness[1], sd.brightness[2], sd.brightness[3]]).astype(np.float64).T.copy()

		# Antennas
		A0=np.int64(np.random.rand(nbl)*na)
		A1=np.int64(np.random.rand(nbl)*na)

		# Create a the jones matrices, but make them identity matrices
		Sols=np.ones((nsrc,nchan,na))[:,:,:,np.newaxis]*(np.eye(2).reshape(4)).astype(np.complex128)

		# Matrix containing information, here just the reference frequency
		# to estimate the flux from spectral index
		Info=np.array([1e6],np.float64)

		# Call Cyrils' predict code
		predict_start = time.time()
		P1=predict.predictSolsPol(Vis, A0, A1, uvw, lms, WaveL, Sols, Info)
		predict_end = time.time()

		log.debug('predict start: %fs end: %fs elapsed time: %fs',
			predict_start, predict_end, predict_end - predict_start)

		return Vis
	
	def do_predict_test(self, shared_data, log):
		sd = shared_data		
		# Choose between the double and float kernel
		rime_bk = RimeJonesBK() if sd.ft == np.float64 else RimeJonesBKFloat()

		# Initialise the node
		rime_bk.initialise(sd)

		vis_predict = self.time_predict(sd, log)

		log.debug('jones_gpu size: %.2f MB', sd.jones_gpu.nbytes/(1024*1024))

		# Set up the segmented reduction
		# Create the key positions. This snippet creates an array
		# equal to the list of positions of the last array element timestep)
		keys = (np.arange(np.product(sd.jones_shape[:-1]))*sd.jones_shape[-1])\
			.astype(np.int32).reshape(sd.jones_shape[:-2])
		
		# Send the keys to the gpu, and create the output array for
		# the visibilities.
		keys_gpu = gpuarray.to_gpu(keys)
		vis_gpu = gpuarray.empty(shape=keys.shape, dtype=sd.jones_gpu.dtype.type)

		bk_params = rime_bk.get_kernel_params(sd)

		kernels_start, kernels_end = cuda.Event(), cuda.Event()
		kernels_start.record()

		# Invoke the kernel
		rime_bk.execute(sd)

		# Choose between the double and float kernel
		if sd.ft == np.float64:
			crimes.segmented_reduce_complex128_sum(
				data=sd.jones_gpu, seg_starts=keys_gpu, seg_sums=vis_gpu,
				device_id=0)
		else:
			crimes.segmented_reduce_complex64_sum(
				data=sd.jones_gpu, seg_starts=keys_gpu, seg_sums=vis_gpu,
				device_id=0)

		kernels_end.record()
		kernels_end.synchronize()

		log.debug('kernels: elapsed time: %fs',
			kernels_start.time_till(kernels_end)*1e-3)

		# Shutdown the rime node, we don't need it any more
		rime_bk.shutdown(sd)

		# Shift the gpu jones matrices so they are on the last axis
		vis_cpu = np.rollaxis(vis_gpu.get(),0,len(sd.jones_shape)-2)

		# Compare the GPU solution with Cyril's predict code
		self.assertTrue(np.allclose(vis_cpu, vis_predict))


	def test_predict_double(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=1,nsrc=10000)
		log = logging.getLogger('TestRimes.test_predict_double')
		self.do_predict_test(sd, log)

	def test_predict_float(self):
		sd = TestRimeSharedData(na=10,nchan=32,ntime=1,nsrc=10000,dtype=np.float32)
		log = logging.getLogger('TestRimes.test_predict_float')
		self.do_predict_test(sd, log)

if __name__ == '__main__':
	logging.basicConfig(stream=sys.stderr)
	logging.getLogger('TestRimes.test_predict_double').setLevel(logging.DEBUG)
	logging.getLogger('TestRimes.test_predict_float').setLevel(logging.DEBUG)

	suite = unittest.TestLoader().loadTestsFromTestCase(TestRimes)
	unittest.TextTestRunner(verbosity=2).run(suite)