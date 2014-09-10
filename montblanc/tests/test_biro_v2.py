import logging
import unittest
import numpy as np
import time
import sys

import montblanc.ext.crimes
import montblanc.ext.predict

from montblanc.impl.biro.v2.TestSharedData import TestSharedData

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum

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

        # This beam width produces reasonable values
        # for testing the E term
        sd.set_beam_width(65*1e5)

        rime_ek = RimeEK()
        rime_cpu = RimeCPU(sd)

        # Initialise the BK float kernel
        rime_ek.initialise(sd)
        rime_ek.execute(sd)
        rime_ek.shutdown(sd)

        ek_cpu = rime_cpu.compute_ek_jones_scalar_per_ant()
        ek_gpu = sd.jones_scalar_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(ek_cpu, ek_gpu,**cmp))

    def test_EK_float(self):
        """ Single precision EK test """
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,dtype=np.float32)      

        self.EK_test_impl(sd)

    def test_EK_double(self):
        """ Double precision EK test """
        sd = TestSharedData(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,dtype=np.float64)      

        self.EK_test_impl(sd)

    def gauss_B_sum_test_impl(self, sd, weight_vector=False, cmp=None):
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        sd.set_beam_width(65*1e5)
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
        sd = TestSharedData(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20,
            dtype=np.float32)      

        self.gauss_B_sum_test_impl(sd, weight_vector=False, cmp={'rtol' : 1e-3})
        self.gauss_B_sum_test_impl(sd, weight_vector=True, cmp={'rtol' : 1e-3})

    def test_gauss_B_sum_double(self):
        """ """
        sd = TestSharedData(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20,
            dtype=np.float64)      

        self.gauss_B_sum_test_impl(sd, weight_vector=False)
        self.gauss_B_sum_test_impl(sd, weight_vector=True)

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

        # Determine the per baseline UVW coordinates
        # from the per antenna UVW coordinates
        ant0, ant1 = sd.get_flat_ap_idx()
        uvw_per_ant = sd.uvw_cpu.reshape(3,sd.ntime*sd.na)
        uvw_per_bl = (uvw_per_ant[:,ant1] - uvw_per_ant[:,ant0])\
            .astype(np.float64).T.copy()

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
            Vis, A0, A1, uvw_per_bl, lms, WaveL, Sols, Info)
        predict_end = time.time()

        return Vis
    
    def do_predict_test(self, shared_data):
        sd = shared_data  
        from montblanc.impl.biro.v2.cpu.RimeCPU import RimeCPU

        # This beam width produces reasonable values
        # for testing the E term
        sd.set_beam_width(65*1e5)        
        sd.transfer_point_errors(np.zeros(
            shape=sd.point_errors_shape,
            dtype=sd.point_errors_dtype))

        rime_cpu = RimeCPU(sd)

        vis_predict_cyril = self.time_predict(sd).transpose(2,0,1)
        vis_predict_cpu = rime_cpu.compute_bk_vis().squeeze()

        # Compare the GPU solution with Cyril's predict code
        self.assertTrue(np.allclose(vis_predict_cyril, vis_predict_cpu))

    def test_predict_double(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64)

        self.do_predict_test(sd)

    def test_predict_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32)

        self.do_predict_test(sd)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)