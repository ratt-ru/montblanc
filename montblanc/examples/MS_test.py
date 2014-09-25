import numpy as np
import montblanc

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-v','--version',dest='version', type=str, default='v1', choices=['v1','v2'],
        help='BIRO Pipeline Version.')        

    args = parser.parse_args(sys.argv[1:])

    # Get the BK pipeline and shared data.
    with montblanc.get_biro_solver(args.msfile,
        npsrc=1, ngsrc=0, dtype=np.float64, version=args.version) as slvr:

        # Create point sources at zeros
        l=slvr.ft(np.ones(slvr.nsrc))*0.1
        m=slvr.ft(np.ones(slvr.nsrc))*0.25
        lm=np.array([l,m], dtype=slvr.ft).reshape(2,slvr.nsrc)

        # Create 1Jy point sources
        fI=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)*2
        fQ=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        fU=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        fV=slvr.ft(np.zeros(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)
        alpha=slvr.ft(np.ones(slvr.ntime*slvr.nsrc)).reshape(slvr.ntime,slvr.nsrc)*0.5
        brightness = np.array([fI,fQ,fU,fV,alpha], dtype=slvr.ft)

        slvr.transfer_lm(lm)
        slvr.transfer_brightness(brightness)

        slvr.solve()

        print slvr.X2