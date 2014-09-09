import logging
import unittest
import numpy as np
import time
import sys

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

#import montblanc
import montblanc.ext.predict
import montblanc.ext.crimes

from montblanc.impl.biro.v1.TestSharedData import TestSharedData

from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
from montblanc.impl.biro.v1.gpu.RimeEBK import RimeEBK
from montblanc.impl.biro.v1.gpu.RimeSumFloat import RimeSumFloat
from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce
from montblanc.impl.biro.v1.gpu.RimeMultiply import RimeMultiply
from montblanc.impl.biro.v1.gpu.RimeChiSquared import RimeChiSquared

from montblanc.impl.biro.v1.cpu.RimeCPU import RimeCPU

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

    def BK_test_impl(self, sd, cmp=None):
        """ Type independent implementation of the BK test """
        if cmp is None: cmp = {}

        rime_bk = RimeBK()

        # Initialise the BK float kernel
        rime_bk.initialise(sd)
        rime_bk.execute(sd)
        rime_bk.shutdown(sd)

        # Compute the jones matrix on the CPU
        jones_cpu = RimeCPU(sd).compute_bk_jones()

        # Get the jones matrices calculated by the GPU
        jones_gpu = sd.jones_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(jones_cpu, jones_gpu,**cmp))

    def test_BK_float(self):
        """ single precision BK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=200,
            dtype=np.float32, device=pycuda.autoinit.device)      

        self.BK_test_impl(sd)

    def test_BK_double(self):
        """ double precision BK test """
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=200,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.BK_test_impl(sd)

    def EBK_test_impl(self,sd,cmp=None):
        """ Type independent implementation of the EBK test """
        if cmp is None: cmp = {}

        sd.set_beam_width(65*1e5)

        kernels = []

        if sd.npsrc > 0: kernels.append(RimeEBK(gaussian=False))
        if sd.ngsrc > 0: kernels.append(RimeEBK(gaussian=True))

        # Invoke the EBK kernels
        for k in kernels: k.initialise(sd)
        for k in kernels: k.execute(sd)
        for k in kernels: k.shutdown(sd)

        jones_gpu = sd.jones_gpu.get()
        jones_cpu = RimeCPU(sd).compute_ebk_jones()

        self.assertTrue(np.allclose(jones_gpu, jones_cpu,**cmp))

    def test_pEBK_double(self):
        """ double precision EBK test for point sources only """
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=200,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.EBK_test_impl(sd)

    def test_pEBK_float(self):
        """ single precision EBK test for point sources only """
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=200,
            dtype=np.float32, device=pycuda.autoinit.device)

        # Hmmm, we don't need this tolerance now? I wonder why it's working...
        #self.EBK_test_impl(sd, cmp={'rtol' : 1e-2,'atol' : 1e-2})
        self.EBK_test_impl(sd)

    def test_pgEBK_double(self):
        """ double precision EBK test for point and gaussian sources """
        #sd = TestSharedData(na=2,nchan=2,ntime=1,npsrc=2,ngsrc=1,
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.EBK_test_impl(sd)

    def test_pgEBK_float(self):
        """ single precision EBK test for point and gaussian sources """
        sd = TestSharedData(na=10,nchan=32,ntime=10,npsrc=100,ngsrc=100,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.EBK_test_impl(sd, cmp={'rtol' : 1e-2})

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

    @unittest.skipIf(False, 'test_multiply_double numpy code is somewhat inefficient')
    def test_multiply_double(self):
        """ double precision multiplication test """
        # Make the problem size smaller, due to slow numpy code in multiply_test_impl
        sd = TestSharedData(na=5,nchan=4,ntime=2,npsrc=10,
            dtype=np.float64, device=pycuda.autoinit.device)
        
        self.multiply_test_impl(sd)

    @unittest.skipIf(False, 'test_multiply_float numpy code is somewhat inefficient')
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

    def test_chi_squared_double(self):
        """ double precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd)

    def test_chi_squared_float(self):
        """ single precision chi squared test """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, cmp={'rtol' : 1e-4})

    def test_chi_squared_weight_vector_double(self):
        """ double precision chi squared test with noise vector """
        sd = TestSharedData(na=20,nchan=32,ntime=100,npsrc=2,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.chi_squared_test_impl(sd, weight_vector=True)

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

    def test_reduce_double(self):
        """ double precision reduction test """
        sd = TestSharedData(na=10, nchan=32, ntime=10, npsrc=200,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.reduce_test_impl(sd)

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

    def test_gauss_double(self):
        """ Gaussian with fwhm and without is the same """
        sd = TestSharedData(na=10, nchan=32, ntime=10, npsrc=10, ngsrc=10,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.gauss_test_impl(sd)

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

    def test_predict_double(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float64, device=pycuda.autoinit.device)

        self.do_predict_test(sd)

    def test_predict_float(self):
        sd = TestSharedData(na=10,nchan=32,ntime=1,npsrc=10000, ngsrc=0,
            dtype=np.float32, device=pycuda.autoinit.device)

        self.do_predict_test(sd)

    def test_sum_float(self):
        sd = TestSharedData(na=14,nchan=32,ntime=36,npsrc=100,
            dtype=np.float32, device=pycuda.autoinit.device)

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

	def demo_real_problem(self):

        """
        Method to demonstrate a pymultinest 'real world' problem
        No pointing errors
        No time variability
        """

        # Settings
        sqrtTwo=sqrt(2.0)
        arcsec2rad = sc.pi / 180.0 / 3600.0
        # Montblanc settings
        loggingLevel=logging.WARN                      # Logging level
        msfile=None # Input MS file
        store_cpu=False         # Carry out the calculation on the CPU
        use_noise_vector=False                         # Varying noise level
        dtype=np.float32                               # or np.float64
        # Sky
        n_params=8
        npsrc=0                                        # no. point sources
        ngsrc=1                                        # no. gaussians
        # Telescope
        tel='WSRT'
        tel_params=dict('WSRT':{'nant':14,'nchan':1}\
                        )
        # Observation
        ntime=72
        uvw_f='uvw_coords.txt'
        sigmaSim=0.1 # Conventional noise per visibility
        sigmaFit=None # If None, fit for sigma, otherwise specify here

        # Multinest settings
        hypo=1 # Run number
        verbose=True # Helpful print statements in the output
        nlive=1000 # Number of live points for MultiNest
        evtol=0.5 # Evidence tolerance for MultiNest
        efr=0.1   # Target sampling efficiency
        resume=False # Resume interrrupted MultiNest runs
        seed=4747 # Random no. generator seed (-ve for system clock)
        ins=False # Use Importance Nested Sampling? (Multinest)
        maxiter=0 # maximum number of iterations for multinest
        multimodal=False
        mode_tolerance=-1e90 # (Beware the old PyMultinest bug)

        # Simulate the target source(s) here
        l_sim=sd.ft(0.0); m_sim=sd.ft(0.0)
        lm_sim=shape_list([l_sim,m_sim], sd.lm_shape, sd.lm_dtype)
        fI_sim=sd.ft(numpy.random.normal(1.0,sigmaSim,ntime*nsrc))

        fQ_sim=sd.ft(np.ones(ntime*nsrc)*0.0) # Unpolarized
        fU_sim=sd.ft(np.ones(ntime*nsrc)*0.0)
        fV_sim=sd.ft(np.ones(ntime*nsrc)*0.0)
        alpha_sim=sd.ft(np.ones(ntime*nsrc)*0.0) # Flat
        brightness_sim = shape_list([fI_sim,fQ_sim,fU_sim,fV_sim,alpha_sim],
            sd.brightness_shape, sd.brightness_dtype)

        el_sim = sd.ft(np.ones(ngsrc)*0.5)
        em_sim = sd.ft(np.ones(ngsrc)*0.5)
        R_sim = sd.ft(np.ones(ngsrc)*100.0)
        gauss_shape = shape_list([el_sim,em_sim,R_sim],
            sd.gauss_shape_shape, sd.gauss_shape_dtype)

        # Generate nchan frequencies/wavelengths
    	frequencies = sd.ft(np.linspace(1e6,2e6,nchan))
        wavelength = sd.ft(montblanc.constants.C/frequencies)
        sd.set_ref_wave(wavelength[nchan//2])

        # Set up the priors
        from priors import Priors
        pri=None
        def myprior(cube, ndim, nparams):
            global pri
            if pri is None: pri=Priors()
            cube[0] = pri.GeneralPrior(cube[0],'U',-720.0*arcsec2rad,720.0*arcsec2rad)
            cube[1] = pri.GeneralPrior(cube[1],'U',-720.0*arcsec2rad,720.0*arcsec2rad)
            cube[2] = pri.GeneralPrior(cube[2],'LOG',1.0e-2,5.0)
            cube[3] = pri.GeneralPrior(cube[3],'U',1.0e-2,1.0)
            cube[4] = pri.GeneralPrior(cube[4],'U',-5.0,5.0)
            cube[5] = pri.GeneralPrior(cube[5],'U',1.0*arcsec2rad,60.0*arcsec2rad)
            cube[6] = pri.GeneralPrior(cube[6],'U',1.0*arcsec2rad,60.0*arcsec2rad)
            cube[7] = pri.GeneralPrior(cube[7],'U',0.0,sc.pi)
            return

        #-----------------------------------------------------------------------

        # And the likelihood
        pipeline=None
        def myloglike(cube, ndim, nparams):
            global pipeline,sd,store_cpu,dtype,use_noise_vector,npsrc,ngsrc,ndata
            # Initialize pipeline once (from file or self-simulation)
            if pipeline is None:
                montblanc.log.setLevel(loggingLevel)
		        sd = BiroSharedData(na=tel_params[tel]['nant'],\
                                    nchan=tel_params[tel]['nchan'],\
                                    ntime=ntime,npsrc=npsrc,ngsrc=ngsrc,\
                     			    dtype=dtype)
        	    uvw = np.genfromtxt(uvw_f)
		        sd.transfer_uvw(uvw)

        		wavelength = ...
		        sd.transfer_wavelength(wavelength)

                pipeline.initialise(sd)
                print sd
                # Carry out calculations on the CPU (as opposed to GPU)
                if store_cpu:
                    point_errors = np.zeros(2*sd.na*sd.ntime)\
                        .astype(sd.ft).reshape((2,sd.na,sd.ntime))
                    sd.transfer_point_errors(point_errors)

                 # Set up the antenna table if noise depends on this
                 if use_noise_vector:
                     bl2ants=sd.get_default_ant_pairs()

                 # Find total number of visibilities
                 ndata=8*sd.nbl*sd.nchan*sd.ntime
            # End of setup

            # Do next lines on every iteration

            # Now set up the model vis for this iteration
            lm=np.array([cube[0],cube[1]], dtype=sd.ft).reshape(sd.lm_shape)

            fI=cube[2]*np.ones((sd.ntime,sd.nsrc,),dtype=sd.ft)
            fQ=cube[4]*np.ones((sd.ntime,sd.nsrc),dtype=sd.ft)
            fU=        np.zeros((sd.ntime,sd.nsrc),dtype=sd.ft)
            fV=        np.zeros((sd.ntime,sd.nsrc),dtype=sd.ft)
            alpha=     np.zeros((sd.ntime,sd.nsrc),dtype=sd.ft)
            brightness = np.array([fI,fQ,fU,fV,alpha],dtype=sd.ft)\
                .reshape(sd.brightness_shape)

            # Push cube parameters for this iteration to the GPU
            sd.transfer_lm(lm)
            sd.transfer_brightness(brightness)

            if sd.ngsrc > 0:
                if cube[5]>cube[6]: return -1.0e99 # Catch oblates(?)
                gauss_shape = np.array([[cube[6]*np.sin(cube[7]),\
                              cube[6]*np.cos(cube[7]),cube[5]/cube[6]]],dtype=sd.ft)\
                              .reshape(sd.gauss_shape_shape)
                sd.transfer_gauss_shape(gauss_shape)

            # Set the noise
            if sigmaFit is not None:
                sigma=sigmaFit*np.ones(1).astype(sd.ft)
            else:
                sigma=cube[3]*np.ones(1).astype(sd.ft)

            if use_noise_vector:
                noise_vector=sigma[0]**2**np.ones(ndata/8).astype(sd.ft)\
                    .reshape(sd.noise_vector_shape)

                sd.transfer_noise_vector(noise_vector)
            else:
                sd.set_sigma_sqrd((sigma[0]**2))

            # Execute the pipeline; cube[:] -> sd.X2
            pipeline.execute(sd)
            if store_cpu:
                chi2=sd.compute_biro_chi_sqrd()
            else:
                chi2=sd.X2

            loglike=-chi2/2.0 - ndata*0.5*np.log(2.0*sc.pi*sigma[0]**2.0)

            return loglike

        #-----------------------------------------------------------------------

        # Now run multinest
        import time
        import pymultinest
        outdir = 'chains-hypo%i' % hypo
        os.mkdir(outdir)
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

        print 'PyMultinest took ', (tend-tstart)/60.0, ' minutes to run\n'
        print 'Avg. execution time %gms' % pipeline.avg_execution_time
        print 'Last execution time %gms' % pipeline.last_execution_time
        print 'No. of executions %d.' % pipeline.nr_of_executions
        print sd

        #-----------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV1)
    unittest.TextTestRunner(verbosity=2).run(suite)
