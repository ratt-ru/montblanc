#!/usr/bin/env python

##########################
# Program to perform Bayesian analysis of visibilities for source identification
##########################

from settings import *
from priors import Priors

# Montblanc
import pycuda.autoinit
import pycuda.driver as cuda
import montblanc
import logging


pipeline=None
def myloglike(cube, ndim, nparams):
    """
    Simple chisq likelihood for straight-line fit (m=1,c=1)

    cube is the unit hypercube containing the current values of parameters
    ndim is the number of dimensions of cube
    nparams (>= ndim) allows extra derived parameters to be carried along
    """
    global pipeline,slvr,store_cpu,dtype,use_noise_vector,npsrc,ngsrc,ndata;

    # Initialize pipeline only once
    if pipeline is None:
        montblanc.log.setLevel(loggingLevel)
        pipeline, slvr = montblanc.get_biro_pipeline(msfile,\
                 npsrc=npsrc,ngsrc=ngsrc,\
                 noise_vector=use_noise_vector,\
                 device=pycuda.autoinit.device,dtype=dtype,\
                 store_cpu=store_cpu)

        pipeline.initialise(slvr)
        print slvr.ft, slvr.ct
        print slvr.vis_shape
        print slvr

        # Carry out calculations on the CPU (as opposed to GPU)
        if store_cpu:
            point_errors = np.zeros(2*slvr.na*slvr.ntime)\
                .astype(slvr.ft).reshape((2, slvr.na, slvr.ntime))
            # And transfer them
            slvr.transfer_point_errors(point_errors)

        # Set up the antenna table if noise depends on this
        if use_noise_vector:
            bl2ants=slvr.get_default_ant_pairs()

        # Find total number of visibilities
        ndata=8*slvr.nbl*slvr.nchan*slvr.ntime

    # End of setup


    # Do next lines on every iteration

    # Now set up the model vis for this iteration
    # For now just do an unpolarized source of unknown flux at the centre
    #cube[0]=0.0; cube[1]=0.0
    #cube[0]=450.0*arcsec2rad
    #cube[1]=600.0*arcsec2rad
    #cube[2]=1.0 # Fix flux
    #cube[4]=0.198089176046084943E-03
    #cube[4]=0.145782635074809264E-04
    #cube[4]=0.0 # Stokes Q flux
    lm=np.array([cube[0],cube[1]], dtype=slvr.ft).reshape(slvr.lm_shape)

    fI=cube[2]*np.ones((slvr.ntime,slvr.nsrc,),dtype=slvr.ft)
    fQ=cube[4]*np.ones((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
    fU=        np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
    fV=        np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
    alpha=     np.zeros((slvr.ntime,slvr.nsrc),dtype=slvr.ft)
    brightness = np.array([fI,fQ,fU,fV,alpha],dtype=slvr.ft)\
        .reshape(slvr.brightness_shape)

    # Push cube parameters for this iteration to the GPU
    slvr.transfer_lm(lm)
    slvr.transfer_brightness(brightness)
    #cube[5]=0.68e-4; cube[6]=0.68e-4
    #cube[7]=0.8
    if slvr.ngsrc > 0:
        # lproj, mproj, ratio
        #gauss_shape=np.array([cube[5],cube[6],cube[7]],dtype=slvr.ft)\
        #    .reshape(slvr.gauss_shape_shape)
        # major, minor, position angle
        # Could this be done by restricting position-angle prior? :
        if cube[5]>cube[6]: return -1.0e99 # Catch oblates(?)
        gauss_shape = np.array([[cube[6]*np.sin(cube[7]),\
                      cube[6]*np.cos(cube[7]),cube[5]/cube[6]]],dtype=slvr.ft)\
            .reshape(slvr.gauss_shape_shape)
        slvr.transfer_gauss_shape(gauss_shape)

    # Set the noise
    if sigmaSim is not None:
        sigma=sigmaSim*np.ones(1).astype(slvr.ft)
    else:
        sigma=cube[3]*np.ones(1).astype(slvr.ft)

    if use_noise_vector:
        noise_vector=sigma[0]**2**np.ones(ndata/8).astype(slvr.ft)\
            .reshape(slvr.noise_vector_shape)

        # Noise as a function of baseline (i.e. antenna)
        #noise_vector=np.zeros(ndata/8).astype(slvr.ft)\
        #    .reshape(slvr.noise_vector_shape) # (nbl,nchan,ntime)
        # Assume for now sig != sig(t,nu)
        #for ibl in range(slvr.nbl):
        #    ants=bl2ants[:,ibl,0] # (2,nbl,ntime)
        #    noise_vector[ibl,:,:]=sqrt(sigAnt[ants[0]]*sigAnt[ants[1]])
        slvr.transfer_noise_vector(noise_vector)
    else:
        slvr.set_sigma_sqrd((sigma[0]**2))

    # Execute the pipeline; cube[:] -> slvr.X2
    pipeline.execute(slvr)
    if store_cpu:
        chi2=slvr.compute_biro_chi_sqrd()
    else:
        chi2=slvr.X2

    # Here we need to worry about the chisq prefactor
    loglike=-chi2/2.0 - ndata*0.5*np.log(2.0*sc.pi*sigma[0]**2.0)

    return loglike

#------------------------------------------------------------------------------

pri=None
def myprior(cube, ndim, nparams):
    """
    This function just transforms parameters to the unit hypercube

    cube is the unit hypercube containing the current values of parameters
    ndim is the number of dimensions of cube
    nparams (>= ndim) allows extra derived parameters to be carried along

    You can use Priors from priors.py for convenience functions:

    from priors import Priors
    pri=Priors()
    cube[0]=pri.UniformPrior(cube[0],x1,x2)
    cube[1]=pri.GaussianPrior(cube[1],mu,sigma)
    cube[2]=pri.DeltaFunctionPrior(cube[2],x1,anything_ignored)
    """

    # Need to convert RA, Dec to dra, ddec
    #ra0 = 0.0; dec0 = 60.0; # user-specified (in degrees)
    #ra0 = ra0 * deg2rad;
    #dec0 = dec0 * deg2rad;
    #ra = ra0 - cube[1]; dec = dec0 + cube[2];
    global pri
    if pri is None: pri=Priors()

    # x, y, S, sigma, Q, lproj, mproj, ratio
    cube[0] = pri.GeneralPrior(cube[0],'U',-720.0*arcsec2rad,720.0*arcsec2rad)
    cube[1] = pri.GeneralPrior(cube[1],'U',-720.0*arcsec2rad,720.0*arcsec2rad)
    cube[2] = pri.GeneralPrior(cube[2],'LOG',1.0e-2,5.0)
    cube[3] = pri.GeneralPrior(cube[3],'U',1.0e-2,1.0)
    cube[4] = pri.GeneralPrior(cube[4],'U',-5.0,5.0)
    #cube[5] = pri.GeneralPrior(cube[5],'U',0.0,60.0*arcsec2rad)
    #cube[6] = pri.GeneralPrior(cube[6],'U',0.0,60.0*arcsec2rad)
    #cube[7] = pri.GeneralPrior(cube[7],'U',0.1,1.5)
    cube[5] = pri.GeneralPrior(cube[5],'U',1.0*arcsec2rad,60.0*arcsec2rad)
    cube[6] = pri.GeneralPrior(cube[6],'U',1.0*arcsec2rad,60.0*arcsec2rad)
    cube[7] = pri.GeneralPrior(cube[7],'U',0.0,sc.pi)

    return

#------------------------------------------------------------------------------

def main():
    #global msreadflag,ra0,dec0,ndata;

    #if initmpi:
    rank = MPI.COMM_WORLD.Get_rank();
    size = MPI.COMM_WORLD.Get_size();
    if rank==0:
      print "Starting execution...";
      print "From process %d: MPI using %d processes..."%(rank, size);

    outdir = 'chains-hypo%i' % hypo;
    if rank == 0 and not os.path.exists(outdir):
      os.mkdir(outdir);

    if hypo < 0 or hypo > 100:
        print '*** WARNING: Illegal hypothesis... exiting!'
        return 1

    print 'hypo ================ %d'%hypo

    outstem = os.path.join(outdir,'%s-'%hypo)


    if verbose and rank==0:
	print "Calling PyMultinest..."
        tstart = time.time();

    pymultinest.run(myloglike,myprior,n_params,evidence_tolerance=evtol,resume=resume,verbose=True,multimodal=multimodal,write_output=True,\
    mode_tolerance=mode_tolerance,n_live_points=nlive,outputfiles_basename=outstem,init_MPI=True,importance_nested_sampling=ins,seed=seed,\
    sampling_efficiency=efr,max_iter=maxiter);

    if verbose and rank==0:
        tend = time.time();

        print "PyMultinest took ", (tend-tstart)/60.0, " minutes to run.\n"
        print 'Avg. execution time %gms' % pipeline.avg_execution_time
        print 'Last execution time %gms' % pipeline.last_execution_time
        print 'No. of executions %d.' % pipeline.nr_of_executions
        print slvr

    return 0

if __name__ == '__main__':

    ret=main()
    sys.exit(ret)
