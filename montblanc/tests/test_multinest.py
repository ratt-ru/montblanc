import logging
import unittest
import numpy as np
import time
import sys

import montblanc.ext.crimes

import montblanc.factory

class TestMultinest(unittest.TestCase):
    """
    Class thats tests multinest
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)
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

    @unittest.skip('Most people won''t have multinest installed')
    def test_multinest(self):
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
                    dtype=mb_params['dtype'], \
                    weight_vector=mb_params['use_weight_vector'],\
                    store_cpu=mb_params['store_cpu'])

        print slvr

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
        slvr.solve()

        # Fetch the simulated visibilities back and add noise
        vis_sim=slvr.vis_gpu.get()
        np.random.seed(noise_seed)
        noise_sim=slvr.ct((np.random.normal(0.0,sigmaSim,4*slvr.nbl*slvr.nchan*slvr.ntime*slvr.nsrc)+1j*np.random.normal(0.0,sigmaSim,4*slvr.nbl*slvr.nchan*slvr.ntime*slvr.nsrc)).reshape(slvr.vis_shape))
        vis_sim+=noise_sim
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
        sampler['slvr']=None
        sampler['ndata']=None
        def myloglike(cube, ndim, nparams):
            # Initialize the solver
            if sampler['slvr'] is None:
                sampler['slvr']=factory.get_biro_solver(sd_type='biro',\
                    na=tel_params[tel]['nant'],\
                    nchan=tel_params[tel]['nchan'],ntime=obs_params['ntime'],\
                    npsrc=sky_params['npsrc'],ngsrc=sky_params['ngsrc'],\
                    dtype=mb_params['dtype'],\
                    weight_vector=mb_params['use_weight_vector'],\
                    store_cpu=mb_params['store_cpu'])
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

                sampler['slvr'].initialise()
                print slvr
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
            sampler['slvr'].solve()
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
        print 'Avg. execution time %gms' % sampler['slvr'].pipeline.avg_execution_time
        print 'Last execution time %gms' % sampler['slvr'].pipeline.last_execution_time
        print 'No. of executions %d.' % sampler['slvr'].pipeline.nr_of_executions
        print sample['slvr']
        # We use this because we can't use context managers with the
        # paradigm used to obtained the solver above
        sampler['slvr'].shutdown()
        print source_params
        lproj_sim=source_params['emaj']*np.sin(source_params['pa'])
        mproj_sim=source_params['emaj']*np.cos(source_params['pa'])
        ratio_sim=source_params['emin']/source_params['emaj']
        print lproj_sim
        print mproj_sim
        print ratio_sim

        #-----------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultinest)
    unittest.TextTestRunner(verbosity=2).run(suite)