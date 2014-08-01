import logging
import unittest
import numpy as np
import time
import sys

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc.ext.predict
import montblanc.ext.crimes

from montblanc.impl.biro.v2.TestSharedData import TestSharedData

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum

from montblanc.impl.biro.v2.gpu.RimeJonesReduce import RimeJonesReduce
from montblanc.impl.biro.v2.gpu.RimeMultiply import RimeMultiply
from montblanc.impl.biro.v2.gpu.RimeChiSquared import RimeChiSquared

from montblanc.impl.biro.v2.cpu.RimeCPU import RimeCPU

class TestBiroV2(unittest.TestCase):
    """
    TestRimes class defining the unit test cases for montblanc
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()*100))
        # Set up various things that aren't possible in PyCUDA
        montblanc.ext.crimes.setup_cuda()

        # Add a handler that outputs INFO level logging
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)

        montblanc.log.addHandler(fh)
        montblanc.log.setLevel(logging.INFO)

    def tearDown(self):
        """ Tear down each test case """
        pass

    def EK_test_impl(self, sd, cmp=None):
        """ Type independent implementaiton of the EK test """
        if cmp is None: cmp = {}

        rime_ek = RimeEK()
        rime_cpu = RimeCPU(sd)

        # Initialise the BK float kernel
        rime_ek.initialise(sd)
        rime_ek.execute(sd)
        rime_ek.shutdown(sd)

        jones_scalar_cpu = rime_cpu.compute_ek_jones_scalar_per_ant()
        jones_scalar_gpu = sd.jones_scalar_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(jones_scalar_cpu, jones_scalar_gpu,**cmp))

    def test_EK_float(self):
        """ Single precision EK test """
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=10,
            dtype=np.float32, device=pycuda.autoinit.device)      

        self.EK_test_impl(sd)

    def test_EK_double(self):
        """ Double precision EK test """
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=10,
            dtype=np.float64, device=pycuda.autoinit.device)      

        self.EK_test_impl(sd)

    def gauss_B_sum_test_impl(self, sd, cmp=None):
        if cmp is None: cmp = {}

        rime_ek = RimeEK()
        rime_gauss_B_sum = RimeGaussBSum()
        rime_cpu = RimeCPU(sd)

        kernels = [rime_gauss_B_sum]

        for k in kernels: k.initialise(sd)
        for k in kernels: k.execute(sd)
        for k in kernels: k.shutdown(sd)

        ebk_vis_cpu = rime_cpu.compute_ebk_vis()
        ebk_vis_gpu = sd.vis_gpu.get()

        self.assertTrue(np.allclose(ebk_vis_cpu, ebk_vis_gpu, **cmp))

    def test_gauss_B_sum_float(self):
        """ """
        sd = TestSharedData(na=10,nchan=64,ntime=96,npsrc=50,ngsrc=50,
        #sd = TestSharedData(na=5,nchan=4,ntime=4,npsrc=2,ngsrc=2,
            dtype=np.float32, device=pycuda.autoinit.device)      

#        self.gauss_B_sum_test_impl(sd, cmp={'rtol' : 1e-5, 'atol' : 1e-8})
        self.gauss_B_sum_test_impl(sd)

    def test_gauss_B_sum_double(self):
        """ """
        sd = TestSharedData(na=10,nchan=64,ntime=96,npsrc=50,ngsrc=50,
            dtype=np.float64, device=pycuda.autoinit.device)      

        self.gauss_B_sum_test_impl(sd)


    def multiply_test_impl(self, sd, cmp=None):
        """ Type independent implementation of the multiply test """
        if cmp is None: cmp = {}

        rime_multiply = RimeMultiply()

        rime_multiply.initialise(sd)

        # Output jones matrix
        njones = sd.nbl*sd.npsrc*sd.nchan*sd.ntime
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
                    for src in range(sd.npsrc):
                        jones_output_cpu[:,bl,ch,t,src] = np.dot(
                        jones_lhs[:,bl,ch,t,src].reshape(2,2),
                        jones_rhs[:,bl,ch,t,src].reshape(2,2)).reshape(4)

        # Confirm similar results
        self.assertTrue(np.allclose(jones_output_gpu, jones_output_cpu,**cmp))

    @unittest.skipIf(True, 'test_multiply_double numpy code is somewhat inefficient')
    def test_multiply_double(self):
        """ double precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,npsrc=10,
            dtype=np.float64, device=pycuda.autoinit.device)
        
        self.multiply_test_impl(sd)

    @unittest.skipIf(True, 'test_multiply_float numpy code is somewhat inefficient')
    def test_multiply_float(self):
        """ single precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,npsrc=10,
            dtype=np.float32, device=pycuda.autoinit.device)
        
        self.multiply_test_impl(sd)

    def chi_squared_test_impl(self, sd, weight_vector=False, cmp=None):
        """ Type independent implementation of the chi squared test """
        if cmp is None: cmp = {}

        kernels = [RimeChiSquared(weight_vector=weight_vector)]

        # Initialise, execute and shutdown the kernels
        for k in kernels: k.initialise(sd)
        for k in kernels: k.execute(sd)
        for k in kernels: k.shutdown(sd)

        # Compute the chi squared sum values
        chi_sqrd_cpu = RimeCPU(sd).compute_chi_sqrd_sum_terms(weight_vector=weight_vector)
        chi_sqrd_gpu = sd.chi_sqrd_result_gpu.get()

        # Check the values inside the sum term of the Chi Squared
        self.assertTrue(np.allclose(chi_sqrd_cpu, chi_sqrd_gpu,**cmp))

        # Compute the actual chi squared value
        X2_cpu = RimeCPU(sd).compute_chi_sqrd(weight_vector=weight_vector)

        # Check that the result returned by the CPU and GPU are the same,
        self.assertTrue(np.allclose(np.array([X2_cpu]), np.array([sd.X2]), **cmp))

    @unittest.skip('ignore for now')
    def test_chi_squared_double(self):
        """ double precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd)

    @unittest.skip('ignore for now')
    def test_chi_squared_float(self):
        """ single precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, cmp={'rtol' : 1e-4})

    @unittest.skip('ignore for now')
    def test_chi_squared_weight_vector_double(self):
        """ double precision chi squared test with noise vector """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, weight_vector=True)

    @unittest.skip('ignore for now')
    def test_chi_squared_weight_vector_float(self):
        """ single precision chi squared test with noise vector """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, weight_vector=True, cmp={'rtol' : 1e-3 })

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

    @unittest.skip('ignore for now')
    def test_reduce_double(self):
        """ double precision reduction test """
        sd = TestSharedData(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.reduce_test_impl(sd)

    @unittest.skip('ignore for now')
    def test_reduce_float(self):
        """ single precision reduction test """
        sd = TestSharedData(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.reduce_test_impl(sd)

    def gauss_test_impl(self, shared_data, cmp=None):
        """ Type independent implementation of the gaussian tests """
        if cmp is None: cmp = {}

        sd = shared_data

        gs = RimeCPU(sd).compute_gaussian_shape()
        gs_with_fwhm = RimeCPU(sd).compute_gaussian_shape_with_fwhm()

        self.assertTrue(np.allclose(gs,gs_with_fwhm, **cmp))

    @unittest.skip('ignore for now')
    def test_gauss_double(self):
        """ Gaussian with fwhm and without is the same """
        sd = TestSharedData(na=10, nchan=32, ntime=10, npsrc=10, ngsrc=10,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.gauss_test_impl(sd)

    @unittest.skip('ignore for now')
    def test_gauss_float(self):
        """ Gaussian with fwhm and without is the same """
        sd = TestSharedData(na=10, nchan=32, ntime=10,npsrc=10, ngsrc=10, \
            dtype=np.float32, device=pycuda.autoinit.device)

        self.gauss_test_impl(sd, cmp={'rtol' : 1e-4 })

    def time_predict(self, sd):
        na=sd.na          # Number of antenna
        nbl=sd.nbl        # Number of baselines
        nchan=sd.nchan    # Number of channels
        nsrc=sd.nsrc      # Number of sources
        ntime=sd.ntime    # Number of timesteps

        # We only handle a timestep of 1 here, because the predict code doesn't handle time
        assert ntime == 1

        # Visibilities ! has to have double complex
        Vis=np.complex128(np.zeros((nbl,nchan,4)))
        # UVW coordinates
        uvw=sd.uvw_cpu.T.astype(np.float64).copy()

        # Frequencies in Hz
        WaveL = sd.wavelength_cpu.astype(np.float64)
        # Sky coordinates
        T=0 # We should only have one timestep in this test

        lms=np.array([sd.lm_cpu[0], sd.lm_cpu[1],
            sd.brightness_cpu[0,T], sd.brightness_cpu[4,T], sd.brightness_cpu[1,T],
            sd.brightness_cpu[2,T], sd.brightness_cpu[3,T]]).astype(np.float64).T.copy()

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

        montblanc.log.info('predict start: %fs end: %fs elapsed time: %fs',
            predict_start, predict_end, predict_end - predict_start)

        return Vis
    
    def do_predict_test(self, shared_data):
        sd = shared_data        
        rime_bk = RimeBK()
        rime_reduce = RimeJonesReduce()

        # Initialise the node
        rime_bk.initialise(sd)

        vis_predict_cpu = self.time_predict(sd)

        montblanc.log.info(sd)

        kernels_start, kernels_end = cuda.Event(), cuda.Event()
        kernels_start.record()

        # Invoke the kernel
        rime_bk.execute(sd)
        rime_reduce.execute(sd)

        kernels_end.record()
        kernels_end.synchronize()

        montblanc.log.info('kernels: elapsed time: %fs',
            kernels_start.time_till(kernels_end)*1e-3)

        # Shutdown the rime node, we don't need it any more
        rime_bk.shutdown(sd)

        # Shift the gpu jones matrices so they are on the last axis
        vis_gpu = np.rollaxis(sd.vis_gpu.get(),0,len(sd.jones_shape)-2).squeeze()

        # Compare the GPU solution with Cyril's predict code
        self.assertTrue(np.allclose(vis_gpu, vis_predict_cpu))

    @unittest.skip('ignore for now')
    def test_predict_double(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.do_predict_test(sd)

    @unittest.skip('ignore for now')
    def test_predict_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.do_predict_test(sd)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)