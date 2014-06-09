import logging
import unittest
import numpy as np
import time
import sys

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc
import montblanc.ext.predict
import montblanc.ext.crimes

from montblanc.RimeBK import RimeBK
from montblanc.RimeEBK import RimeEBK
from montblanc.RimeEBKSumFloat import RimeEBKSumFloat
from montblanc.RimeSumFloat import RimeSumFloat
from montblanc.RimeJonesReduce import RimeJonesReduce
from montblanc.RimeMultiply import RimeMultiply
from montblanc.RimeChiSquared import RimeChiSquared
from montblanc.RimeChiSquaredReduce import RimeChiSquaredReduce
from montblanc.TestSharedData import TestSharedData

class TestRimes(unittest.TestCase):
    """
    TestRimes class defining the unit test cases for montblanc
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()*100))
        # Set up various things that aren't possible in PyCUDA
        montblanc.ext.crimes.setup_cuda()

        # Add a handler that outputs DEBUG level logging
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.DEBUG)

        montblanc.log.addHandler(fh)
        montblanc.log.setLevel(logging.DEBUG)

    def tearDown(self):
        """ Tear down each test case """
        pass

    def BK_test_impl(self, sd, cmp=None):
        """ Type independent implementation of the BK test """
        if cmp is None: cmp = {}

        rime_bk = RimeBK()

        # Initialise the BK float kernel
        rime_bk.initialise(sd)
        rime_bk.execute(sd)
        rime_bk.shutdown(sd)

        # Compute the jones matrix on the CPU
        jones_cpu = sd.compute_bk_jones()

        # Get the jones matrices calculated by the GPU
        jones_gpu = sd.jones_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(jones_cpu, jones_gpu,**cmp))

    def test_BK_float(self):
        """ single precision BK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,dtype=np.float32,
            device=pycuda.autoinit.device)      

        self.BK_test_impl(sd)

    def test_BK_double(self):
        """ double precision BK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,
            device=pycuda.autoinit.device)

        self.BK_test_impl(sd)

    def EBK_test_impl(self,sd,cmp=None):
        """ Type independent implementation of the EBK test """
        if cmp is None: cmp = {}

        sd.set_beam_width(65*1e5)

        rime_ebk = RimeEBK()

        # Invoke the EBK kernel
        rime_ebk.initialise(sd)
        rime_ebk.execute(sd)        
        rime_ebk.shutdown(sd)

        jones_gpu = sd.jones_gpu.get()
        jones_cpu = sd.compute_ebk_jones()

        self.assertTrue(np.allclose(jones_gpu, jones_cpu,**cmp))

    def test_EBK_double(self):
        """ double precision EBK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,
            device=pycuda.autoinit.device)

        self.EBK_test_impl(sd)

    def test_EBK_float(self):
        """ single precision EBK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,dtype=np.float32,
            device=pycuda.autoinit.device)

        # Hmmm, we don't need this tolerance now? I wonder why it's working...
        #self.EBK_test_impl(sd, cmp={'rtol' : 1e-2,'atol' : 1e-2})
        self.EBK_test_impl(sd)

    def multiply_test_impl(self, sd, cmp=None):
        """ Type independent implementation of the multiply test """
        if cmp is None: cmp = {}

        rime_multiply = RimeMultiply()

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
        self.assertTrue(np.allclose(jones_output_gpu, jones_output_cpu,**cmp))

    @unittest.skipIf(False, 'test_multiply_double numpy code is somewhat inefficient')
    def test_multiply_double(self):
        """ double precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,nsrc=10,
            device=pycuda.autoinit.device)
        
        self.multiply_test_impl(sd)

    @unittest.skipIf(False, 'test_multiply_float numpy code is somewhat inefficient')
    def test_multiply_float(self):
        """ single precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,nsrc=10,
            device=pycuda.autoinit.device)
        
        self.multiply_test_impl(sd)

    def chi_squared_test_impl(self, sd, noise_vector=False, cmp=None):
        """ Type independent implementation of the chi squared test """
        if cmp is None: cmp = {}

        kernels = [RimeChiSquared(), RimeChiSquaredReduce(noise_vector=noise_vector)]

        # Initialise, execute and shutdown the kernels
        for k in kernels: k.initialise(sd)
        for k in kernels: k.execute(sd)
        for k in kernels: k.shutdown(sd)

        # Compute the chi squared sum values
        chi_sqrd_cpu = sd.compute_chi_sqrd_sum_terms()
        chi_sqrd_gpu = sd.chi_sqrd_result_gpu.get()

        # Check the values inside the sum term of the Chi Squared
        self.assertTrue(np.allclose(chi_sqrd_cpu, chi_sqrd_gpu,**cmp))

        # Compute the actual chi squared value
        X2_cpu = sd.compute_chi_sqrd(noise_vector=noise_vector)

        # Check that the result returned by the CPU and GPU are the same,
        self.assertTrue(np.allclose(np.array([X2_cpu]), np.array([sd.X2]), **cmp))

    def test_chi_squared_double(self):
        """ double precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,nsrc=2,
            device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd)

    def test_chi_squared_float(self):
        """ single precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,nsrc=2,dtype=np.float32,
            device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, cmp={'rtol' : 1e-4})

    def test_chi_squared_noise_vector_double(self):
        """ double precision chi squared test with noise vector """
        sd = TestSharedData(na=20,nchan=32,ntime=100,nsrc=2,
            device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, noise_vector=True)

    def test_chi_squared_noise_vector_float(self):
        """ single precision chi squared test with noise vector """
        sd = TestSharedData(na=20,nchan=32,ntime=100,nsrc=2,dtype=np.float32,
            device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, noise_vector=True, cmp={'rtol' : 1e-3 })

    def reduce_test_impl(self, sd, cmp=None):
        """ Type independent implementation of the reduction tests """
        if cmp is None: cmp = {}

        rime_reduce = RimeJonesReduce()

        rime_reduce.initialise(sd)

        # Create the jones matrices
        jsize = np.product(sd.jones_shape)
        jones_cpu = (np.random.random(jsize) + 1j*np.random.random(jsize))\
            .astype(sd.ct).reshape(sd.jones_shape)
    
        # Send the jones and keys to the gpu, and create the output array for
        # the visibilities.
        sd.transfer_jones(jones_cpu)

        # Compute the reduction
        rime_reduce.execute(sd)

        # Shutdown the rime node, we don't need it any more
        rime_reduce.shutdown(sd)

        # Add everything along the last axis (time)
        vis_cpu = np.sum(jones_cpu,axis=len(sd.jones_shape)-1)

        # Get the visibilities off the GPU
        vis_gpu = sd.vis_gpu.get()

        # Confirm similar results
        self.assertTrue(np.allclose(vis_cpu, vis_gpu, **cmp))

    def test_reduce_double(self):
        """ double precision reduction test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,
            device=pycuda.autoinit.device)

        self.reduce_test_impl(sd)

    def test_reduce_float(self):
        """ single precision reduction test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=200,dtype=np.float32,
            device=pycuda.autoinit.device)

        self.reduce_test_impl(sd)

    def time_predict(self, sd):
        na=sd.na          # Number of antenna
        nbl=sd.nbl        # Number of baselines
        nchan=sd.nchan    # Number of channels
        nsrc=sd.nsrc      # Number of sources
        ntime=sd.ntime    # Number of timesteps

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
        Info=np.array([sd.ref_wave],np.float64)

        # Call Cyrils' predict code
        predict_start = time.time()
        P1=montblanc.ext.predict.predictSolsPol(
            Vis, A0, A1, uvw, lms, WaveL, Sols, Info)
        predict_end = time.time()

        montblanc.log.debug('predict start: %fs end: %fs elapsed time: %fs',
            predict_start, predict_end, predict_end - predict_start)

        return Vis
    
    def do_predict_test(self, shared_data):
        sd = shared_data        
        rime_bk = RimeBK()
        rime_reduce = RimeJonesReduce()

        # Initialise the node
        rime_bk.initialise(sd)

        vis_predict_cpu = self.time_predict(sd)

        montblanc.log.debug(sd)

        kernels_start, kernels_end = cuda.Event(), cuda.Event()
        kernels_start.record()

        # Invoke the kernel
        rime_bk.execute(sd)
        rime_reduce.execute(sd)

        kernels_end.record()
        kernels_end.synchronize()

        montblanc.log.debug('kernels: elapsed time: %fs',
            kernels_start.time_till(kernels_end)*1e-3)

        # Shutdown the rime node, we don't need it any more
        rime_bk.shutdown(sd)

        # Shift the gpu jones matrices so they are on the last axis
        vis_gpu = np.rollaxis(sd.vis_gpu.get(),0,len(sd.jones_shape)-2).squeeze()

        # Compare the GPU solution with Cyril's predict code
        self.assertTrue(np.allclose(vis_gpu, vis_predict_cpu))

    def test_predict_double(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,nsrc=10000,
            device=pycuda.autoinit.device)

        self.do_predict_test(sd)

    def test_predict_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,nsrc=10000,dtype=np.float32,
            device=pycuda.autoinit.device)

        self.do_predict_test(sd)

    def test_sum_float(self):
        sd = TestSharedData(na=14,nchan=32,ntime=36,nsrc=100,dtype=np.float32,
            device=pycuda.autoinit.device)
        jones_cpu = (np.random.random(np.product(sd.jones_shape)) + \
            np.random.random(np.product(sd.jones_shape))*1j)\
            .reshape(sd.jones_shape).astype(sd.ct)
        sd.jones_gpu.set(jones_cpu)
        
        rime_sum = RimeSumFloat()
        rime_sum.initialise(sd)

        rime_sum.execute(sd)

        vis_cpu = np.add.reduce(jones_cpu,axis=4)

        self.assertTrue(np.allclose(vis_cpu, sd.vis_gpu.get()))

        rime_sum.shutdown(sd)

    def test_EBK_sum_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=10,nsrc=100,dtype=np.float32,
            device=pycuda.autoinit.device)
        rime_ebk_sum = RimeEBKSumFloat()

        # Initialise the BK float kernel
        rime_ebk_sum.initialise(sd)
        rime_ebk_sum.execute(sd)
        rime_ebk_sum.shutdown(sd)

        # Compute the jones matrix on the CPU, and sum over
        # the sources (axis 4)
        vis_cpu = sd.compute_bk_vis()

        # Get the visibilities calculated by the GPU
        vis_gpu = sd.vis_gpu.get()

        # Test that the CPU calculation matches that of the GPU calculation
        # By default, rtol=1e-5. The different summation orders on the CPU
        # and GPU seem to make a difference here, so we relax the tolerance
        # somewhat.
        self.assertTrue(np.allclose(vis_cpu, vis_gpu, rtol=1e-4))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRimes)
    unittest.TextTestRunner(verbosity=2).run(suite)