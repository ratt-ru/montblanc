import logging
import unittest
import numpy as np
import time
import sys

import montblanc.ext.predict
import montblanc.ext.crimes

from montblanc.impl.biro.v1.TestSolver import TestSolver

from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
from montblanc.impl.biro.v1.gpu.RimeEBK import RimeEBK
from montblanc.impl.biro.v1.gpu.RimeSumFloat import RimeSumFloat
from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce
from montblanc.impl.biro.v1.gpu.RimeMultiply import RimeMultiply
from montblanc.impl.biro.v1.gpu.RimeChiSquared import RimeChiSquared

from montblanc.impl.biro.v1.cpu.RimeCPU import RimeCPU

from montblanc.pipeline import Pipeline

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
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=200,
            pipeline=Pipeline([RimeBK()]), dtype=np.float32) as slvr:

            self.BK_test_impl(slvr)

    def test_BK_double(self):
        """ double precision BK test """
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=200,
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
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=200,ngsrc=0,
            dtype=np.float64, pipeline=Pipeline([RimeEBK(gaussian=False)])) as slvr:

            self.EBK_test_impl(slvr)

    def test_pEBK_float(self):
        """ single precision EBK test for point sources only """
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=200,ngsrc=0,
            dtype=np.float32, pipeline=Pipeline([RimeEBK(gaussian=False)])) as slvr:

            # Hmmm, we don't need this tolerance now? I wonder why it's working...
            #self.EBK_test_impl(slvr, cmp={'rtol' : 1e-2,'atol' : 1e-2})
            self.EBK_test_impl(slvr)

    def test_pgEBK_double(self):
        """ double precision EBK test for point and gaussian sources """
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
            dtype=np.float64,
            pipeline=Pipeline([RimeEBK(gaussian=False), RimeEBK(gaussian=True)])) as slvr:

            self.EBK_test_impl(slvr)

    def test_pgEBK_float(self):
        """ single precision EBK test for point and gaussian sources """
        with TestSolver(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
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
        with TestSolver(na=5,nchan=4,ntime=2,npsrc=10,dtype=np.float64) as slvr:
        
            self.multiply_test_impl(slvr)

    @unittest.skipIf(False, 'test_multiply_float numpy code is somewhat inefficient')
    def test_multiply_float(self):
        """ single precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        with TestSolver(na=5,nchan=4,ntime=2,npsrc=10,dtype=np.float32) as slvr:
        
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
        with TestSolver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=False)])) as slvr:

            self.chi_squared_test_impl(slvr)

    def test_chi_squared_float(self):
        """ single precision chi squared test """
        with TestSolver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=False)])) as slvr:

            self.chi_squared_test_impl(slvr, cmp={'rtol' : 1e-4})

    def test_chi_squared_weight_vector_double(self):
        """ double precision chi squared test with noise vector """
        with TestSolver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float64,
            pipeline=Pipeline([RimeChiSquared(weight_vector=True)])) as slvr:

            self.chi_squared_test_impl(slvr, weight_vector=True)

    def test_chi_squared_weight_vector_float(self):
        """ single precision chi squared test with noise vector """
        with TestSolver(na=20,nchan=32,ntime=100,npsrc=2,dtype=np.float32,
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
        with TestSolver(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float64, pipeline=Pipeline([RimeJonesReduce()])) as slvr:

            self.reduce_test_impl(slvr)

    def test_reduce_float(self):
        """ single precision reduction test """
        with TestSolver(na=10, nchan=32, ntime=10, npsrc=200,
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
        slvr = TestSolver(na=10,nchan=32,ntime=10,npsrc=10,ngsrc=10,dtype=np.float64)

        self.gauss_test_impl(slvr)

    def test_gauss_float(self):
        """ Gaussian with fwhm and without is the same """
        slvr = TestSolver(na=10,nchan=32,ntime=10,npsrc=10,ngsrc=10,dtype=np.float32)

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
        with TestSolver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64, pipeline=Pipeline([RimeBK(), RimeJonesReduce()])) as slvr:

            self.predict_test_impl(slvr)

    def test_predict_float(self):
        with TestSolver(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32, pipeline=Pipeline([RimeBK(), RimeJonesReduce()])) as slvr:

            self.predict_test_impl(slvr)

    def test_sum_float(self):
        slvr = TestSolver(na=14,nchan=32,ntime=36,npsrc=100,dtype=np.float32)

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

    def demo_real_problem(self):
        """
        Method to demonstrate a pymultinest 'real world' problem
        No pointing errors
        No time variability
        Run as
        cd montblanc/tests/
        python -m unittest test_biro_v1.TestBiroV1.demo_real_problem
        """

        import os,random
        cuda_device = os.environ.get('CUDA_DEVICE',int(random.getrandbits(1)))
        os.environ['CUDA_DEVICE']=str(cuda_device)
        print 'Using GPU #%i' % int(os.environ.get('CUDA_DEVICE'))

        # Settings
        import montblanc.factory as factory
        import scipy as sc
        sqrtTwo=np.sqrt(2.0)
        arcsec2rad = sc.pi / 180.0 / 3600.0
        deg2rad = sc.pi / 180.0

        # Montblanc settings
        loggingLevel=logging.WARN                      # Logging level
        mb_params={'store_cpu':False,'use_weight_vector':False,'dtype':np.float32}
        # Sky
        params_to_fit=['x','y','I','alpha','lproj','mproj','ratio','sigma']
        n_params=len(params_to_fit)
        sky_params={'npsrc':0,'ngsrc':1}
        source_params={'I':1.0,'Q':0.0,'U':0.0,'V':0.0,\
                               'x':-80.0*arcsec2rad,'y':-100.0*arcsec2rad,\
                               'alpha':-0.7,\
                               'emin':10.0*arcsec2rad,\
                               'emaj':20.0*arcsec2rad,\
                               'pa':45.0*deg2rad}
        # Telescope
        tel='WSRT'
        tel_params={'WSRT':{'nant':14,'nchan':8,'freq':1.4e9,'BW':8*16.0e6},\
                        'ATCA':{'nant':-1,'nchan':-1}}
        # Observation
        obs_params={'ntime':72}
        uvw_f='uvw_coords.txt'
        sigmaSim=0.1 # Conventional noise per visibility
        sigmaFit=None # If None, fit for sigma, otherwise specify here
        noise_seed=1234

        # MultiNest settings
        hypo=1 # Run number
        verbose=True # Helpful print statements in the output
        nlive=1000 # Number of live points
        evtol=0.5 # Evidence tolerance
        efr=0.1   # Target sampling efficiency
        resume=False # Resume interrrupted runs
        seed=4747 # Random no. generator seed (-ve for system clock)
        ins=False # Use Importance Nested Sampling?
        maxiter=0 # maximum number of iterations
        multimodal=False
        mode_tolerance=-1e90 # (Beware the old PyMultinest bug)

        # http://stackoverflow.com/questions/11987358/why-nested-functions-can-access-variables-from-outer-functions-but-are-not-allo
        sampler={} # This is a mutable to permit semi-globals(!)

        #-----------------------------------------------------------------------

        # Simulate some data
        montblanc.log.setLevel(loggingLevel)
        slvr=factory.get_biro_solver(sd_type='biro',\
                    na=tel_params[tel]['nant'],\
                    nchan=tel_params[tel]['nchan'],ntime=obs_params['ntime'],\
                    npsrc=sky_params['npsrc'],ngsrc=sky_params['ngsrc'],\
                    dtype=mb_params['dtype'])
        sampler['simulator'] = factory.get_biro_pipeline(\
                    npsrc=sky_params['npsrc'],ngsrc=sky_params['ngsrc'],\
                    weight_vector=mb_params['use_weight_vector'],\
                    dtype=mb_params['dtype'],\
                    store_cpu=mb_params['store_cpu'])

        print slvr
	#print sampler['simulator']

        # Set up the observation
        uvw = slvr.ft(np.genfromtxt(uvw_f).T.reshape((3,-1,obs_params['ntime'])))
        slvr.transfer_uvw(uvw)

        frequencies = slvr.ft(np.linspace(tel_params[tel]['freq'],tel_params[tel]['freq']+tel_params[tel]['BW'],slvr.nchan))
        #frequencies = slvr.ft(np.linspace(1.4e9,1.5e9,slvr.nchan))
        wavelength = slvr.ft(montblanc.constants.C/frequencies)
        slvr.set_ref_wave(wavelength[slvr.nchan//2])
        slvr.transfer_wavelength(wavelength)

        # Simulate the target source(s) here
        lm_sim=slvr.ft(np.array([source_params['x'],source_params['y']]).reshape(slvr.lm_shape))
        slvr.transfer_lm(lm_sim)
        fI_sim=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)*source_params['I'])
        fQ_sim=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)*source_params['Q'])
        fU_sim=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)*source_params['U'])
        fV_sim=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)*source_params['V'])
        alpha_sim=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)*source_params['alpha'])
        brightness_sim=np.array([fI_sim,fQ_sim,fU_sim,fV_sim,alpha_sim]).reshape(slvr.brightness_shape)
        slvr.transfer_brightness(brightness_sim)

        e1=source_params['emaj']*np.sin(source_params['pa'])
        e2=source_params['emaj']*np.cos(source_params['pa'])
        r=source_params['emin']/source_params['emaj']
        el_sim = slvr.ft(np.ones(slvr.ngsrc)*e1)
        em_sim = slvr.ft(np.ones(slvr.ngsrc)*e2)
        R_sim = slvr.ft(np.ones(slvr.ngsrc)*r)
        gauss_shape_sim = np.array([el_sim,em_sim,R_sim]).reshape(slvr.gauss_shape_shape)
        slvr.transfer_gauss_shape(gauss_shape_sim)
        sampler['simulator'].initialise(slvr)
        sampler['simulator'].execute(slvr)

        # Fetch the simulated visibilities back and add noise
        vis_sim=slvr.vis_gpu.get()
        np.random.seed(noise_seed)
        noise_sim=slvr.ct((np.random.normal(0.0,sigmaSim,4*slvr.nbl*slvr.nchan*slvr.ntime*slvr.nsrc)+1j*np.random.normal(0.0,sigmaSim,4*slvr.nbl*slvr.nchan*slvr.ntime*slvr.nsrc)).reshape(slvr.vis_shape))
        vis_sim+=noise_sim
        sampler['simulator'].shutdown(slvr)
        sampler['simulator']=vis_sim
        # End of simulation step

        #-------------------------------------------------------------------

        # Set up the priors
        from montblanc.tests.priors import Priors
        sampler['pri']=None
        def myprior(cube, ndim, nparams):
            if sampler['pri'] is None: sampler['pri']=Priors()
            #cube[0]=source_params['x']
            #cube[1]=source_params['y']
            #cube[2]=source_params['I']
            #cube[3]=sigmaSim
            #cube[4]=source_params['alpha']
            cube[0] = sampler['pri'].GeneralPrior(cube[0],'U',-360.0*arcsec2rad,360.0*arcsec2rad) # x
            cube[1] = sampler['pri'].GeneralPrior(cube[1],'U',-360.0*arcsec2rad,360.0*arcsec2rad) # y
            cube[2] = sampler['pri'].GeneralPrior(cube[2],'LOG',1.0e-2,4.0) # I
            cube[3] = sampler['pri'].GeneralPrior(cube[3],'U',1.0e-2,1.0) # noise
            cube[4] = sampler['pri'].GeneralPrior(cube[4],'U',-3.0,3.0) # alpha
            cube[5] = sampler['pri'].GeneralPrior(cube[5],'U',1.0*arcsec2rad,30.0*arcsec2rad) # lproj
            cube[6] = sampler['pri'].GeneralPrior(cube[6],'U',1.0*arcsec2rad,30.0*arcsec2rad) # mproj
            cube[7] = sampler['pri'].GeneralPrior(cube[7],'U',0.0,1.0) # ratio
            return

        #-------------------------------------------------------------------
        # Now begin the likelihood calculation proper
        sampler['pipeline']=None
        sampler['slvr']=None
        sampler['ndata']=None
        def myloglike(cube, ndim, nparams):
            # Initialize the pipeline
            if sampler['pipeline'] is None:
                sampler['slvr']=factory.get_biro_solver(sd_type='biro',\
                    na=tel_params[tel]['nant'],\
                    nchan=tel_params[tel]['nchan'],ntime=obs_params['ntime'],\
                    npsrc=sky_params['npsrc'],ngsrc=sky_params['ngsrc'],\
                    dtype=mb_params['dtype'])
                sampler['pipeline'] = factory.get_biro_pipeline(\
                    npsrc=sky_params['npsrc'],ngsrc=sky_params['ngsrc'],\
                    weight_vector=mb_params['use_weight_vector'],\
                    dtype=mb_params['dtype'],\
                    store_cpu=mb_params['store_cpu'])
                slvr=sampler['slvr']
                # Set up the observation (again - probably unnecessary)
                uvw = slvr.ft(np.genfromtxt(uvw_f).T.reshape((3,-1,obs_params['ntime'])))
                slvr.transfer_uvw(uvw)

                frequencies = slvr.ft(np.linspace(tel_params[tel]['freq'],tel_params[tel]['freq']+tel_params[tel]['BW'],slvr.nchan))
                #frequencies = slvr.ft(np.linspace(1.4e9,1.5e9,slvr.nchan))
                wavelength = slvr.ft(montblanc.constants.C/frequencies)
                slvr.set_ref_wave(wavelength[slvr.nchan//2])
                slvr.transfer_wavelength(wavelength)

                # Transfer the simulated vis into slvr
                slvr.transfer_bayes_data(sampler['simulator'])

                sampler['pipeline'].initialise(slvr)
                print slvr
                # Carry out calculations on the CPU (as opposed to GPU)
                if mb_params['store_cpu']:
                    point_errors = np.zeros(2*slvr.na*slvr.ntime)\
                        .astype(slvr.ft).reshape((2,slvr.na,slvr.ntime))
                    slvr.transfer_point_errors(point_errors)

                # Set up the antenna table if noise depends on this
                if mb_params['use_weight_vector']:
                     bl2ants=slvr.get_default_ant_pairs()

                # Find total number of visibilities
                # complex x [XX,XY,YX,YY] x nbl x nchan x ntime
                sampler['ndata']=2*4*slvr.nbl*slvr.nchan*slvr.ntime
            # End of setup

            #-------------------------------------------------------------------
            # Do next lines on every iteration

            # Now set up the model vis for this iteration
            slvr=sampler['slvr']; ndata=sampler['ndata']
            lm=np.array([cube[0],cube[1]], dtype=slvr.ft).reshape(slvr.lm_shape)
            fI=cube[2]*np.ones((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
            fQ=        np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
            fU=        np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
            fV=        np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
            alpha=cube[4]*np.ones((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
            brightness = np.array([fI,fQ,fU,fV,alpha],dtype=slvr.ft)\
                .reshape(slvr.brightness_shape)

            # Push cube parameters for this iteration to the GPU
            slvr.transfer_lm(lm)
            slvr.transfer_brightness(brightness)

            if slvr.ngsrc > 0:
                #if cube[5]>cube[6]: return -1.0e99 # Catch oblates(?)
                #gauss_shape = np.array([[cube[6]*np.sin(cube[7]),\
                #              cube[6]*np.cos(cube[7]),cube[5]/cube[6]]],dtype=slvr.ft).reshape(slvr.gauss_shape_shape)
                gauss_shape = np.array([cube[5],cube[6],cube[7]],dtype=slvr.ft).reshape(slvr.gauss_shape_shape)
                slvr.transfer_gauss_shape(gauss_shape)

            # Set the noise
            if sigmaFit is not None:
                sigma=sigmaFit*np.ones(1).astype(slvr.ft)
            else:
                sigma=cube[3]*np.ones(1).astype(slvr.ft)

            if mb_params['use_weight_vector']:
                # One weight element per set of polarizations
                weight_vector=np.ones(ndata/8).astype(slvr.ft)\
                    .reshape(slvr.weight_vector_shape)/sigma[0]**2

                slvr.transfer_weight_vector(weight_vector)
            else:
                slvr.set_sigma_sqrd((sigma[0]**2))

            # Execute the pipeline; cube[:] -> slvr.X2
            sampler['pipeline'].execute(slvr)
            if mb_params['store_cpu']:
                chi2=slvr.compute_biro_chi_sqrd()
            else:
                chi2=slvr.X2

            if mb_params['use_weight_vector']:
                # I think there needs to be a factor of 8 here because
                # weight_vector is per complex Stokes vector
                loglike=-chi2/2.0 - 8*0.5*np.log(2.0*sc.pi/weight_vector).sum()
            else:
                loglike=-chi2/2.0 - ndata*0.5*np.log(2.0*sc.pi*sigma[0]**2.0)

            return loglike

        #-----------------------------------------------------------------------

        # Now run multinest
        import os,time
        from mpi4py import MPI # This is necessary simply to init (unused) MPI
        import pymultinest
        outdir = 'chains-hypo%i' % hypo
        if not os.path.exists(outdir): os.mkdir(outdir)
        outstem = os.path.join(outdir,'%s-'%hypo)

        print 'Calling PyMultinest...'
        tstart = time.time()
        pymultinest.run(myloglike,myprior,n_params,evidence_tolerance=evtol,\
                        resume=resume,verbose=True,multimodal=multimodal,\
                        write_output=True,mode_tolerance=mode_tolerance,\
                        n_live_points=nlive,outputfiles_basename=outstem,\
                        init_MPI=False,importance_nested_sampling=ins,seed=seed,\
                        sampling_efficiency=efr,max_iter=maxiter)
        tend = time.time()

        #-----------------------------------------------------------------------
        print 'PyMultinest took ', (tend-tstart)/60.0, ' minutes to run\n'
        print 'Avg. execution time %gms' % sampler['pipeline'].avg_execution_time
        print 'Last execution time %gms' % sampler['pipeline'].last_execution_time
        print 'No. of executions %d.' % sampler['pipeline'].nr_of_executions
        print slvr
        sampler['pipeline'].shutdown(slvr)
        print source_params
        lproj_sim=source_params['emaj']*np.sin(source_params['pa'])
        mproj_sim=source_params['emaj']*np.cos(source_params['pa'])
        ratio_sim=source_params['emin']/source_params['emaj']
        print lproj_sim
        print mproj_sim
        print ratio_sim

        #-----------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV1)
    unittest.TextTestRunner(verbosity=2).run(suite)
