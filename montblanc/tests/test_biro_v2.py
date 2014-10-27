import logging
import unittest
import numpy as np
import time
import sys

import montblanc.ext.crimes
import montblanc.ext.predict

import montblanc.factory

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum

from montblanc.impl.biro.v2.cpu.RimeCPU import RimeCPU
from montblanc.pipeline import Pipeline

def solver(**kwargs):
    return montblanc.factory.get_biro_solver('test',version='v2',**kwargs)

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

    def EK_test_impl(self, slvr, cmp=None):
        """ Type independent implementaiton of the EK test """
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)

        rime_cpu = RimeCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ek_cpu = rime_cpu.compute_ek_jones_scalar_per_ant()
        ek_gpu = slvr.jones_scalar_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(ek_cpu, ek_gpu, **cmp))

    def test_EK_float(self):
        """ Single precision EK test """
        with solver(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,
            dtype=np.float32,pipeline=Pipeline([RimeEK()])) as slvr:

            self.EK_test_impl(slvr)

    def test_EK_double(self):
        """ Double precision EK test """
        with solver(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,
            dtype=np.float64,pipeline=Pipeline([RimeEK()])) as slvr:

            self.EK_test_impl(slvr)

    def gauss_B_sum_test_impl(self, slvr, weight_vector=False, cmp=None):
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)
        slvr.set_sigma_sqrd(np.random.random(1)[0])

        rime_cpu = RimeCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ebk_vis_cpu = rime_cpu.compute_ebk_vis()
        ebk_vis_gpu = slvr.vis_gpu.get()

        self.assertTrue(np.allclose(ebk_vis_cpu, ebk_vis_gpu, **cmp))

        chi_sqrd_result_cpu = rime_cpu.compute_biro_chi_sqrd(weight_vector=weight_vector)

        self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))        

    def test_gauss_B_sum_float(self):
        """ """
        for w in [True,False]:
            with solver(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20, dtype=np.float32,
                pipeline=Pipeline([RimeEK(), RimeGaussBSum(weight_vector=w)])) as slvr:

                self.gauss_B_sum_test_impl(slvr, weight_vector=w, cmp={ 'rtol' : 1e-1})
        
    def test_gauss_B_sum_double(self):
        """ """
        for w in [True,False]:
            with solver(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20, dtype=np.float32,
                pipeline=Pipeline([RimeEK(), RimeGaussBSum(weight_vector=w)])) as slvr:

                self.gauss_B_sum_test_impl(slvr, weight_vector=w, cmp={ 'rtol' : 1e-1})

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

        # Determine the per baseline UVW coordinates
        # from the per antenna UVW coordinates
        ant0, ant1 = slvr.get_flat_ap_idx()
        uvw_per_ant = slvr.uvw_cpu.reshape(3,slvr.ntime*slvr.na)
        uvw_per_bl = (uvw_per_ant[:,ant1] - uvw_per_ant[:,ant0])\
            .astype(np.float64).T.copy()

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
            Vis, A0, A1, uvw_per_bl, lms, WaveL, Sols, Info)
        predict_end = time.time()

        return Vis
    
    def do_predict_test(self, solver):
        slvr = solver  
        from montblanc.impl.biro.v2.cpu.RimeCPU import RimeCPU

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)        
        slvr.transfer_point_errors(np.zeros(
            shape=slvr.point_errors_shape,
            dtype=slvr.point_errors_dtype))

        rime_cpu = RimeCPU(slvr)

        vis_predict_cyril = self.time_predict(slvr).transpose(2,0,1)
        vis_predict_cpu = rime_cpu.compute_bk_vis().squeeze()

        # Compare the GPU solution with Cyril's predict code
        self.assertTrue(np.allclose(vis_predict_cyril, vis_predict_cpu))

    def test_predict_double(self):
        with solver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64) as slvr:

            self.do_predict_test(slvr)

    def test_predict_float(self):
        with solver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32) as slvr:

            self.do_predict_test(slvr)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)