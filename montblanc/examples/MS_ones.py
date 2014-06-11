import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import montblanc

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=1, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')

    args = parser.parse_args(sys.argv[1:])

    # Get the BK pipeline and shared data.
    pipeline, sd = montblanc.get_bk_pipeline(args.msfile, npsrc=args.npsrc, ngsrc=args.ngsrc,
        device=pycuda.autoinit.device)

    # Initialise the pipeline
    pipeline.initialise(sd)

    # Create point sources at zeros
    l=sd.ft(np.zeros(sd.npsrc))
    m=sd.ft(np.zeros(sd.npsrc))
    lm=np.array([l,m], dtype=sd.ft)

    # Create 1Jy point sources
    fI=sd.ft(np.ones(sd.npsrc))
    fQ=sd.ft(np.zeros(sd.npsrc))
    fU=sd.ft(np.zeros(sd.npsrc))
    fV=sd.ft(np.zeros(sd.npsrc))
    alpha=sd.ft(np.zeros(sd.npsrc))
    brightness = np.array([fI,fQ,fU,fV,alpha], dtype=sd.ft)

    sd.transfer_lm(lm)
    sd.transfer_brightness(brightness)

    pipeline.execute(sd)

    pipeline.shutdown(sd)

    print sd.vis_gpu.get()

    print 'lm', lm
    print 'brightness', brightness
    print sd.ref_wave
    print sd
