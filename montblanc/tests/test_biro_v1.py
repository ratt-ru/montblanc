import logging
import unittest
import numpy as np
import time
import sys

import montblanc.ext.predict
import montblanc.ext.crimes

import montblanc.factory

from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
from montblanc.impl.biro.v1.gpu.RimeEBK import RimeEBK
from montblanc.impl.biro.v1.gpu.RimeSumFloat import RimeSumFloat
from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce
from montblanc.impl.biro.v1.gpu.RimeMultiply import RimeMultiply
from montblanc.impl.biro.v1.gpu.RimeChiSquared import RimeChiSquared

from montblanc.impl.biro.v1.cpu.RimeCPU import RimeCPU

from montblanc.pipeline import Pipeline

def solver(**kwargs):
    return montblanc.factory.get_biro_solver('test',version='v1',**kwargs)

class TestBiroV1(unittest.TestCase):
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

    def BK_test_impl(self, slvr, cmp=None):
        """ Type independent implementation of the BK test """
        if cmp is None: cmp = {}

        # Call the GPU solver
        slvr.solve()

        # Compute the jones matrix on the CPU
        jones_cpu = RimeCPU(slvr).compute_bk_jones()

        # Get the jones matrices calculated by the GPU
        jones_gpu = slvr.jones_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(jones_cpu, jones_gpu,**cmp))

    def test_BK_float(self):
        """ single precision BK test """
        with solver(na=10,nchan=32,ntime=10,npsrc=200,
            pipeline=Pipeline([RimeBK()]), dtype=np.float32) as slvr:

            self.BK_test_impl(slvr)

    def test_BK_double(self):
        """ double precision BK test """
        with solver(na=10,nchan=32,ntime=10,npsrc=200,
            pipeline=Pipeline([RimeBK()]), dtype=np.float64) as slvr:

            self.BK_test_impl(slvr)

    def EBK_test_impl(self,slvr,cmp=None):
        """ Type independent implementation of the EBK test """
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)

        # Call the GPU solver
        slvr.solve()

        jones_gpu = slvr.jones_gpu.get()
        jones_cpu = RimeCPU(slvr).compute_ebk_jones()

        self.assertTrue(np.allclose(jones_gpu, jones_cpu,**cmp))

    def test_pEBK_double(self):
        """ double precision EBK test for point sources only """
        with solver(na=10,nchan=32,ntime=10,npsrc=200,ngsrc=0,
            dtype=np.float64, pipeline=Pipeline([RimeEBK(gaussian=False)])) as slvr:

            self.EBK_test_impl(slvr)

    def test_pEBK_float(self):
        """ single precision EBK test for point sources only """
        with solver(na=10,nchan=32,ntime=10,npsrc=200,ngsrc=0,
            dtype=np.float32, pipeline=Pipeline([RimeEBK(gaussian=False)])) as slvr:

            # Hmmm, we don't need this tolerance now? I wonder why it's working...
            #self.EBK_test_impl(slvr, cmp={'rtol' : 1e-2,'atol' : 1e-2})
            self.EBK_test_impl(slvr)

    def test_pgEBK_double(self):
        """ double precision EBK test for point and gaussian sources """
        with solver(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
            dtype=np.float64,
            pipeline=Pipeline([RimeEBK(gaussian=False), RimeEBK(gaussian=True)])) as slvr:

            self.EBK_test_impl(slvr)

    def test_pgEBK_float(self):
        """ single precision EBK test for point and gaussian sources """
        with solver(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
            dtype=np.float64,
            pipeline=Pipeline([RimeEBK(gaussian=False), RimeEBK(gaussian=True)])) as slvr:
    
            self.EBK_test_impl(slvr, cmp={'rtol' : 1e-2})

    def multiply_test_impl(self, slvr, cmp=None):
        """ Type independent implementation of the multiply test """
        import pycuda.gpuarray as gpuarray

        if cmp is None: cmp = {}

        rime_multiply = RimeMultiply()

        rime_multiply.initialise(slvr)

        # Output jones matrix
        njones = slvr.nbl*slvr.npsrc*slvr.nchan*slvr.ntime
        jsize = np.product(slvr.jones_shape) # Number of complex  numbers
        # Create some random jones matrices to multiply together
        jones_lhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
            .astype(slvr.ct).reshape(slvr.jones_shape)
        jones_rhs = (np.random.random(jsize) + 1j*np.random.random(jsize))\
            .astype(slvr.ct).reshape(slvr.jones_shape)

        jones_lhs_gpu = gpuarray.to_gpu(jones_lhs)
        jones_rhs_gpu = gpuarray.to_gpu(jones_rhs)

        rime_multiply.kernel(jones_lhs_gpu, jones_rhs_gpu, slvr.jones_gpu,
            np.int32(njones), **rime_multiply.get_kernel_params(slvr))

        # Shutdown the rime node, we don't need it any more
        rime_multiply.shutdown(slvr)

        # Get the result off the gpu
        jones_output_gpu = slvr.jones_gpu.get()

        # Perform the calculation on the CPU
        jones_output_cpu = np.empty(shape=slvr.jones_shape, dtype=slvr.ct)

        # TODO: There must be a more numpy way to do this
        # Its dog slow...
        # Possible alternative to use with np.rollaxis:
        # from numpy.core.umath_tests import matrix_multiply
        # Doesn't work with complex numbers tho
        for bl in range(slvr.nbl):
            for ch in range(slvr.nchan):
                for t in range(slvr.ntime):               
                    for src in range(slvr.npsrc):
                        jones_output_cpu[:,bl,ch,t,src] = np.dot(
                        jones_lhs[:,bl,ch,t,src].reshape(2,2),
                        jones_rhs[:,bl,ch,t,src].reshape(2,2)).reshape(4)

        # Confirm similar results
        self.assertTrue(np.allclose(jones_output_gpu, jones_output_cpu,**cmp))

    @unittest.skipIf(False, 'test_multiply_double numpy code is somewhat inefficient')
    def test_multiply_double(self):
        """ double precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        with solver(na=5,nchan=4,ntime=2,npsrc=10,dtype=np.float64) as slvr:
        
            self.multiply_test_impl(slvr)

    @unittest.skipIf(False, 'test_multiply_float numpy code is somewhat inefficient')
    def test_multiply_float(self):
        """ single precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        with solver(na=5,nchan=4,ntime=2,npsrc=10,dtype=np.float32) as slvr:
        
            self.multiply_test_impl(slvr)

    def chi_squared_test_impl(self, slvr, weight_vector=False, cmp=None):
        """ Type independent implementation of the chi squared test """
        if cmp is None: cmp = {}

        # Call the GPU solver
        slvr.solve()

        # Compute the chi squared sum values
        chi_sqrd_cpu = RimeCPU(slvr).compute_chi_sqrd_sum_terms(weight_vector=weight_vector)
        chi_sqrd_gpu = slvr.chi_sqrd_result_gpu.get()

        # Check the values inside the sum term of the Chi Squared
        self.assertTrue(np.allclose(chi_sqrd_cpu, chi_sqrd_gpu,**cmp))

        # Compute the actual chi squared value
        X2_cpu = RimeCPU(slvr).compute_chi_sqrd(weight_vector=weight_vector)

        # Check that the result returned by the CPU and GPU are the same,
        self.assertTrue(np.allclose(np.array([X2_cpu]), np.array([slvr.X2]), **cmp))

    def test_chi_squared_double(self):
        """ double precision chi squared test """
        with solver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=False)])) as slvr:

            self.chi_squared_test_impl(slvr)

    def test_chi_squared_float(self):
        """ single precision chi squared test """
        with solver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=False)])) as slvr:

            self.chi_squared_test_impl(slvr, cmp={'rtol' : 1e-4})

    def test_chi_squared_weight_vector_double(self):
        """ double precision chi squared test with noise vector """
        with solver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=True)])) as slvr:

            self.chi_squared_test_impl(slvr, weight_vector=True)

    def test_chi_squared_weight_vector_float(self):
        """ single precision chi squared test with noise vector """
        with solver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float32,
            pipeline=Pipeline([RimeChiSquared(weight_vector=True)])) as slvr:

            self.chi_squared_test_impl(slvr,
                weight_vector=True, cmp={'rtol' : 1e-3 })

    def reduce_test_impl(self, slvr, cmp=None):
        """ Type independent implementation of the reduction tests """
        if cmp is None: cmp = {}

        # Create the jones matrices
        jsize = np.product(slvr.jones_shape)
        jones_cpu = (np.random.random(jsize) + 1j*np.random.random(jsize))\
            .astype(slvr.ct).reshape(slvr.jones_shape)
    
        # Send the jones and keys to the gpu, and create the output array for
        # the visibilities.
        slvr.transfer_jones(jones_cpu)

        # Call the GPU solver
        slvr.solve()

        # Add everything along the last axis (time)
        vis_cpu = np.sum(jones_cpu,axis=len(slvr.jones_shape)-1)

        # Get the visibilities off the GPU
        vis_gpu = slvr.vis_gpu.get()

        # Confirm similar results
        self.assertTrue(np.allclose(vis_cpu, vis_gpu, **cmp))

    def test_reduce_double(self):
        """ double precision reduction test """
        with solver(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float64, pipeline=Pipeline([RimeJonesReduce()])) as slvr:

            self.reduce_test_impl(slvr)

    def test_reduce_float(self):
        """ single precision reduction test """
        with solver(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float32, pipeline=Pipeline([RimeJonesReduce()])) as slvr:

            self.reduce_test_impl(slvr)

    def gauss_test_impl(self, solver, cmp=None):
        """ Type independent implementation of the gaussian tests """
        if cmp is None: cmp = {}

        slvr = solver

        gs = RimeCPU(slvr).compute_gaussian_shape()
        gs_with_fwhm = RimeCPU(slvr).compute_gaussian_shape_with_fwhm()

        self.assertTrue(np.allclose(gs,gs_with_fwhm, **cmp))

    def test_gauss_double(self):
        """ Gaussian with fwhm and without is the same """
        slvr = solver(na=10,nchan=32,ntime=10,npsrc=10,ngsrc=10,dtype=np.float64)

        self.gauss_test_impl(slvr)

    def test_gauss_float(self):
        """ Gaussian with fwhm and without is the same """
        slvr = solver(na=10,nchan=32,ntime=10,npsrc=10,ngsrc=10,dtype=np.float32)

        self.gauss_test_impl(slvr, cmp={'rtol' : 1e-4 })

    def time_predict(self, slvr):
        na=slvr.na          # Number of antenna
        nbl=slvr.nbl        # Number of baselines
        nchan=slvr.nchan    # Number of channels
        nsrc=slvr.nsrc      # Number of sources
        ntime=slvr.ntime    # Number of timesteps

        # We only handle a timestep of 1 here, because the predict code doesn't handle time
        assert ntime == 1

        # Visibilities ! has to have double complex
        Vis=np.complex128(np.zeros((nbl,nchan,4)))
        # UVW coordinates
        uvw=slvr.uvw_cpu.T.astype(np.float64).copy()

        # Frequencies in Hz
        WaveL = slvr.wavelength_cpu.astype(np.float64)
        # Sky coordinates
        T=0 # We should only have one timestep in this test

        lms=np.array([slvr.lm_cpu[0], slvr.lm_cpu[1],
            slvr.brightness_cpu[0,T], slvr.brightness_cpu[4,T], slvr.brightness_cpu[1,T],
            slvr.brightness_cpu[2,T], slvr.brightness_cpu[3,T]]).astype(np.float64).T.copy()

        # Antennas
        A0=np.int64(np.random.rand(nbl)*na)
        A1=np.int64(np.random.rand(nbl)*na)

        # Create a the jones matrices, but make them identity matrices
        Sols=np.ones((nsrc,nchan,na))[:,:,:,np.newaxis]*(np.eye(2).reshape(4)).astype(np.complex128)

        # Matrix containing information, here just the reference frequency
        # to estimate the flux from spectral index
        Info=np.array([slvr.ref_wave],np.float64)

        # Call Cyrils' predict code
        predict_start = time.time()
        P1=montblanc.ext.predict.predictSolsPol(
            Vis, A0, A1, uvw, lms, WaveL, Sols, Info)
        predict_end = time.time()

        montblanc.log.info('predict start: %fs end: %fs elapsed time: %fs',
            predict_start, predict_end, predict_end - predict_start)

        return Vis
    
    def predict_test_impl(self, solver):
        import pycuda.driver as cuda

        slvr = solver        

        vis_predict_cpu = self.time_predict(slvr)

        montblanc.log.info(slvr)

        kernels_start, kernels_end = cuda.Event(), cuda.Event()
        kernels_start.record()

        # Call the GPU solver
        slvr.solve()

        kernels_end.record()
        kernels_end.synchronize()

        montblanc.log.info('kernels: elapsed time: %fs',
            kernels_start.time_till(kernels_end)*1e-3)

        # Shift the gpu jones matrices so they are on the last axis
        vis_gpu = np.rollaxis(slvr.vis_gpu.get(),0,len(slvr.jones_shape)-2).squeeze()

        # Compare the GPU solution with Cyril's predict code
        self.assertTrue(np.allclose(vis_gpu, vis_predict_cpu))

    def test_predict_double(self):
        with solver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64, pipeline=Pipeline([RimeBK(), RimeJonesReduce()])) as slvr:

            self.predict_test_impl(slvr)

    def test_predict_float(self):
        with solver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32, pipeline=Pipeline([RimeBK(), RimeJonesReduce()])) as slvr:

            self.predict_test_impl(slvr)

    def test_sum_float(self):
        slvr = solver(na=14,nchan=32,ntime=36,npsrc=100,dtype=np.float32)

        jones_cpu = (np.random.random(np.product(slvr.jones_shape)) + \
            np.random.random(np.product(slvr.jones_shape))*1j)\
            .reshape(slvr.jones_shape).astype(slvr.ct)
        slvr.jones_gpu.set(jones_cpu)
        
        rime_sum = RimeSumFloat()
        rime_sum.initialise(slvr)

        rime_sum.execute(slvr)

        vis_cpu = np.add.reduce(jones_cpu,axis=4)

        self.assertTrue(np.allclose(vis_cpu, slvr.vis_gpu.get()))

        rime_sum.shutdown(slvr)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV1)
    unittest.TextTestRunner(verbosity=2).run(suite)
