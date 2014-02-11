import unittest
from pipedrimes import RimeShared
from RimeJonesBK import RimeJonesBK
from RimeJonesReduce import RimeJonesReduce
from RimeJonesMultiplyInbuilt import RimeJonesMultiply
import numpy as np

class TestRimes(unittest.TestCase):
	def setUp(self):
		sd = self.shared_data = RimeShared()
		sd.configure()

		self.rime_bk = RimeJonesBK()
		self.rime_multiply = RimeJonesMultiply()
		self.rime_reduce = RimeJonesReduce()

	def tearDown(self):
		del self.rime_bk
		del self.rime_multiply
		del self.rime_reduce

	def test_multiply(self):
		import pycuda.autoinit
		import pycuda.gpuarray as gpuarray

		sd, rime_multiply = self.shared_data, self.rime_multiply

		rime_multiply.initialise(sd)

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

		print 'block', block, 'grid', grid

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

		rime_multiply.shutdown(sd)

	def test_BK(self):
		import pycuda.autoinit
		import pycuda.gpuarray as gpuarray

		sd, rime_bk = self.shared_data, self.rime_bk

		rime_bk.initialise(sd)

		freqs=np.float64(np.linspace(1e6,2e6,sd.nchan))
		wavelength = 3e8/freqs

		baselines_per_block = 8 if sd.nbl > 8 else sd.nbl
		srcs_per_block = 128 if sd.nsrc > 128 else sd.nsrc

		baseline_blocks = (sd.nbl + baselines_per_block - 1) / baselines_per_block
		src_blocks = (sd.nsrc + srcs_per_block - 1) / srcs_per_block

		block=(baselines_per_block,srcs_per_block,1)
		grid=(baseline_blocks,src_blocks,1)

		print 'block', block, 'grid', grid

		chan = 0

		rime_bk.kernel(sd.uvw_gpu, sd.lma_gpu, sd.sky_gpu,
		    wavelength[chan],  sd.jones_gpu,
		    np.int32(sd.nsrc), np.int32(sd.na), np.int32(sd.nbl),
		    block=block, grid=grid,
		    shared=3*(baselines_per_block+srcs_per_block)*np.dtype(np.float64).itemsize)

		rime_bk.shutdown(sd)
	
if __name__ == '__main__':
    unittest.main()
