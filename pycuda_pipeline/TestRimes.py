import unittest
from pipedrimes import RimeShared
from RimeJonesBK import RimeJonesBK
from RimeJonesReduce import RimeJonesReduce
from RimeJonesMultiplyInbuilt import RimeJonesMultiply
import numpy as np

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
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
		srcs_per_block = 128 if sd.nsrc > 128 else sd.nsrc

		baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
		src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block

		block=(baselines_per_block,srcs_per_block,1)
		grid=(baseline_blocks,src_blocks,1)

		#print 'block', block, 'grid', grid

		rime_bk.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
		    sd.wavelength_gpu,  sd.jones_gpu,
		    np.int32(sd.nsrc), np.int32(sd.nbl),
		    block=block, grid=grid,
		    shared=3*(baselines_per_block+srcs_per_block)*np.dtype(np.float64).itemsize)

		# Assuming that we only calculate channel 0 on the GPU.
		# TODO: This *will change
		chan = 0

		# Get the jones matrices calculated by the GPU
		jones = sd.jones_gpu.get()
	
		# n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
		n = np.sqrt(1. - sd.lma[0]**2 - sd.lma[1]**2) - 1.

		# u*l+v*m+w*n. Outer product creates array of dim nbl x nsrcs
		phase = np.outer(sd.uvw[0], sd.lma[0]) + \
			np.outer(sd.uvw[1], sd.lma[1]) + \
			np.outer(sd.uvw[2],n)

		# 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nsrcs
		phase = 2*np.pi*1j*np.sqrt(phase)/sd.wavelength[chan]
		# Dim 1xnsrcs
		power = np.power(1e6/sd.wavelength[chan], sd.lma[2])
		# This works due to broadcast! Dim nbl x nsrcs
		phase_term = power*np.exp(phase)

		# Create the brightness matrix. Dim nsrcs x 4
		sky = np.complex128([
			# fU+fQ + 0j
			sd.sky[0]+sd.sky[3] + 0j,
			# fI + fQ*1j
			sd.sky[1] + 1j*sd.sky[2],
			# fI - fQ*1j
			sd.sky[1] - 1j*sd.sky[2],
			# fU-fQ + 0j
			sd.sky[0]-sd.sky[3] + 0j]).T

		# This works due to broadcast! Multiplies along
		# srcs axis of sky. Dim nbl x nsrcs x 4
		jones_cpu = phase_term[:,:,np.newaxis]*sky

		# Test that the jones CPU calculation matches that of the GPU calculation
		self.assertTrue(np.allclose(jones_cpu.flatten(), jones.flatten()))

	def test_multiply(self):
		sd, rime_multiply = self.shared_data, self.rime_multiply

		## Here I define my data, and my Jones matrices
		na=sd.na          # Number of antenna
		nbl=sd.nbl        # Number of baselines
		nchan=sd.nchan    # Number of channels
		nsrc=sd.nsrc      # Number of DDES

		# Output jones matrix
		njones = nbl*nsrc
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

		for baseline in range(nbl):
		    for direction in range(nsrc):
		        jones_output_cpu[:,baseline,direction] = np.dot(
		            jones_lhs[:,baseline,direction].reshape(2,2),
		            jones_rhs[:,baseline,direction].reshape(2,2)).reshape(4)

		# Confirm similar results
		self.assertTrue(np.allclose(jones_output, jones_output_cpu))

	def test_reduce(self):
		sd, rime_reduce = self.shared_data, self.rime_reduce

		na=sd.na
		nbl=sd.nbl
		nchan=sd.nchan
		nsrc=sd.nsrc

		assert sd.jones_shape == (4, sd.nbl, sd.nsrc)

		# Create the jones matrices
		jsize = np.product(sd.jones_shape)
		#jones = np.arange(jsize).astype(np.complex128).reshape(sd.jones_shape);

		jones = (np.random.random(jsize) + 1j*np.random.random(jsize))\
			.astype(np.complex128).reshape(sd.jones_shape)

		# Create the keys
		keys = (np.arange(np.product(sd.jones_shape[0:2]))*sd.jones_shape[2])\
			.astype(np.int32)
		
		# Send the jones and keys to the gpu, and create the output array for
		# the segmented sums
		jones_gpu = gpuarray.to_gpu(jones.flatten())
		keys_gpu = gpuarray.to_gpu(keys)
		sums_gpu = gpuarray.empty(shape=keys.shape, dtype=jones.dtype.type)

		segreduce.segmented_reduce_complex128_sum(
			data=jones_gpu, seg_starts=keys_gpu, seg_sums=sums_gpu,
			device_id=0)

		# Add everything along the src axis
		sums_cpu = np.sum(jones,axis=2)

		# Confirm similar results
		self.assertTrue(np.allclose(sums_cpu.flatten(), sums_gpu.get()))

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(TestRimes)
	unittest.TextTestRunner(verbosity=2).run(suite)