import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import montblanc

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-n','--nsrc',dest='nsrc', type=int, default=17, help='Number of Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')

    args = parser.parse_args(sys.argv[1:])

    # Get the BIRO pipeline and shared data.
    pipeline, sd = montblanc.get_biro_pipeline(args.msfile, nsrc=args.nsrc,
        device=pycuda.autoinit.device)

    # Initialise the pipeline
    pipeline.initialise(sd)

    # Random point source coordinates in the l,m,n (brightness image) domain
    l=sd.ft(np.random.random(sd.nsrc)*0.1)
    m=sd.ft(np.random.random(sd.nsrc)*0.1)
    lm=np.array([l,m], dtype=sd.ft)

    # Random brightness matrix for the point sources
    fI=sd.ft(np.ones((sd.nsrc,)))
    fQ=sd.ft(np.random.random(sd.nsrc)*0.5)
    fU=sd.ft(np.random.random(sd.nsrc)*0.5)
    fV=sd.ft(np.random.random(sd.nsrc)*0.5)
    alpha=sd.ft(np.random.random(sd.nsrc)*0.1)
    brightness = np.array([fI,fQ,fU,fV,alpha], dtype=sd.ft)

    # Create a bayesian model and upload it to the GPU
    nviselements = np.product(sd.vis_shape)
    bayes_data = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
        .astype(sd.ct).reshape(sd.vis_shape)
    sd.transfer_bayes_data(bayes_data)

    # Generate random antenna pointing errors
    point_errors = np.random.random(2*sd.na*sd.ntime)\
        .astype(sd.ft).reshape((2, sd.na, sd.ntime))

    kernels_start, kernels_end = cuda.Event(), cuda.Event()
    time_sum = 0.0

    # Execute the pipeline
    for i in range(args.count):
        kernels_start.record()
        # Set data on the shared data object. Uploads to GPU
        sd.transfer_lm(lm)
        sd.transfer_brightness(brightness)
        sd.transfer_point_errors(point_errors)
        # Change parameters for this run
        sd.set_sigma_sqrd((np.random.random(1)**2)[0])
        # Execute the pipeline
        pipeline.execute(sd)
        kernels_end.record()
        kernels_end.synchronize()
        time_sum += kernels_start.time_till(kernels_end)

        # The chi squared result is set on the shared data object
        print 'Chi Squared Value', sd.X2

        # Obtain the visibilities  (slow)
        #V = sd.vis_gpu.get()

    print 'kernels: elapsed time: %gms' %\
        (time_sum / args.count)
    # Print information about the simulation
    print sd

    pipeline.shutdown(sd)
