import logging
import unittest
import numpy as np
import time
import sys

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
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=10,dtype=np.float32)      

        self.EK_test_impl(sd)

    def test_EK_double(self):
        """ Double precision EK test """
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=10,dtype=np.float64)      

        self.EK_test_impl(sd)

    def gauss_B_sum_test_impl(self, sd, weight_vector=False, cmp=None):
        if cmp is None: cmp = {}

        sd.set_sigma_sqrd(np.random.random(1)[0])

        rime_ek = RimeEK()
        rime_gauss_B_sum = RimeGaussBSum(weight_vector=weight_vector)
        rime_cpu = RimeCPU(sd)

        kernels = [rime_ek, rime_gauss_B_sum]

        for k in kernels: k.initialise(sd)
        for k in kernels: k.execute(sd)
        for k in kernels: k.shutdown(sd)

        ebk_vis_cpu = rime_cpu.compute_ebk_vis()
        ebk_vis_gpu = sd.vis_gpu.get()

        self.assertTrue(np.allclose(ebk_vis_cpu, ebk_vis_gpu, **cmp))

        chi_sqrd_result_cpu = rime_cpu.compute_biro_chi_sqrd(weight_vector=weight_vector)

        self.assertTrue(np.allclose(chi_sqrd_result_cpu, sd.X2, **cmp))        

    def test_gauss_B_sum_float(self):
        """ """
        sd = TestSharedData(na=10,nchan=48,ntime=10,npsrc=10,ngsrc=10,
            dtype=np.float32)      

        self.gauss_B_sum_test_impl(sd, weight_vector=False, cmp={'rtol' : 1e-3})
        self.gauss_B_sum_test_impl(sd, weight_vector=True, cmp={'rtol' : 1e-3})

    def test_gauss_B_sum_double(self):
        """ """
        sd = TestSharedData(na=10,nchan=48,ntime=10,npsrc=10,ngsrc=10,
            dtype=np.float64)      

        self.gauss_B_sum_test_impl(sd, weight_vector=False)
        self.gauss_B_sum_test_impl(sd, weight_vector=True)

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
        sd = TestSharedData(na=5,nchan=4,ntime=2,npsrc=10, dtype=np.float64)
        
        self.multiply_test_impl(sd)

    @unittest.skipIf(True, 'test_multiply_float numpy code is somewhat inefficient')
    def test_multiply_float(self):
        """ single precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,npsrc=10, dtype=np.float32)
        
        self.multiply_test_impl(sd)

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
            dtype=np.float64)

        self.do_predict_test(sd)

    @unittest.skip('ignore for now')
    def test_predict_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32)

        self.do_predict_test(sd)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)