import logging
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import montblanc

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')    
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.WARN)

    # Get the BIRO pipeline and shared data.
    # npsrc : number of point sources
    # ngsrc : number of gaussian sources
    # init_weights : (1) None (2) 'sigma' or (3) 'weight'. Either
    #   (1) do not initialise the weight vector, or
    #   (2) initialise from the MS 'SIGMA' tables, or
    #   (3) initialise from the MS 'WEIGHT' tables.
    # weight_vector : indicates whether a weight vector should be used to
    #   compute the chi squared or a single sigma squared value
    # store_cpu : indicates whether copies of the data passed into the
    #   shared data transfer_* methods should be stored on the shared data object
    pipeline, sd = montblanc.get_biro_pipeline(args.msfile,
        npsrc=args.npsrc, ngsrc=args.ngsrc,
        init_weights=None, weight_vector=False,
        store_cpu=False, device=pycuda.autoinit.device)

    # Initialise the pipeline
    pipeline.initialise(sd)

    # Random point source coordinates in the l,m,n (brightness image) domain
    l=sd.ft(np.random.random(sd.nsrc)*0.1)
    m=sd.ft(np.random.random(sd.nsrc)*0.1)
    lm=np.array([l,m], dtype=sd.ft)

    # Random brightness matrix for the point sources
    fI=sd.ft(np.ones((sd.ntime,sd.nsrc,)))
    fQ=sd.ft(np.random.random(sd.ntime*sd.nsrc)*0.5).reshape(sd.ntime,sd.nsrc)
    fU=sd.ft(np.random.random(sd.ntime*sd.nsrc)*0.5).reshape(sd.ntime,sd.nsrc)
    fV=sd.ft(np.random.random(sd.ntime*sd.nsrc)*0.5).reshape(sd.ntime,sd.nsrc)
    alpha=sd.ft(np.random.random(sd.ntime*sd.nsrc)*0.1).reshape(sd.ntime,sd.nsrc)
    brightness = np.array([fI,fQ,fU,fV,alpha], dtype=sd.ft)

    # If there are gaussian sources, create their
    # shape matrix and transfer it.
    if sd.ngsrc > 0:
        el = sd.ft(np.random.random(sd.ngsrc)*0.1)
        em = sd.ft(np.random.random(sd.ngsrc)*0.1)
        R = sd.ft(np.random.random(sd.ngsrc))
        gauss_shape = np.array([el,em,R],dtype=sd.ft)
        sd.transfer_gauss_shape(gauss_shape)

    # Create a bayesian model and upload it to the GPU
    nviselements = np.product(sd.vis_shape)
    bayes_data = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
        .astype(sd.ct).reshape(sd.vis_shape)
    sd.transfer_bayes_data(bayes_data)

    # Generate random antenna pointing errors
    point_errors = np.random.random(np.product(sd.point_errors_shape))\
        .astype(sd.ft).reshape((sd.point_errors_shape))

    # Generate and transfer a noise vector.
    weight_vector = np.random.random(np.product(sd.weight_vector_shape))\
        .astype(sd.ft).reshape((sd.weight_vector_shape))
    sd.transfer_weight_vector(weight_vector)

    # Execute the pipeline
    for i in range(args.count):
        # Set data on the shared data object. Uploads to GPU
        sd.transfer_lm(lm)
        sd.transfer_brightness(brightness)
        sd.transfer_point_errors(point_errors)
        # Change parameters for this run
        sd.set_sigma_sqrd((np.random.random(1)**2)[0])
        # Execute the pipeline
        pipeline.execute(sd)

        # The chi squared result is set on the shared data object
        print 'Chi Squared Value', sd.X2

        # Must use get_biro_pipeline(...,store_cpu=True)
        # for the following to work
        #X2_cpu = sd.compute_biro_chi_sqrd()
        #print 'Chi Squared Value', sd.X2, X2_cpu, np.allclose(sd.X2,X2_cpu, rtol=1e-2)

        # Obtain the visibilities  (slow)
        #V = sd.vis_gpu.get()

    # Print information about the simulation
    print sd
    print 'Pipeline: avg execution time: %gms last execution time: %gms executions: %d. ' %\
        (pipeline.avg_execution_time, pipeline.last_execution_time, pipeline.nr_of_executions)

    pipeline.shutdown(sd)
