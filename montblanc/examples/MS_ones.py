import numpy as np
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
    with montblanc.get_bk_solver(args.msfile, npsrc=args.npsrc, ngsrc=args.ngsrc) as slvr:

        # Create point sources at zeros
        l=slvr.ft(np.zeros(slvr.nsrc))
        m=slvr.ft(np.zeros(slvr.nsrc))
        lm=np.array([l,m], dtype=slvr.ft)

        # Create 1Jy point sources
        fI=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        fQ=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        fU=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        fV=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        alpha=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        brightness = np.array([fI,fQ,fU,fV,alpha], dtype=slvr.ft)

        slvr.transfer_lm(lm)
        slvr.transfer_brightness(brightness)

        # Solve the RIME
        slvr.solve()

        print slvr.vis_gpu.get()
        print slvr
