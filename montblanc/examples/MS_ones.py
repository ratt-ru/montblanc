#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import numpy as np
import montblanc

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=1, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')

    args = parser.parse_args(sys.argv[1:])

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=args.npsrc, gaussian=args.ngsrc, sersic=args.nssrc),
        dtype='double', version=Options.VERSION_TWO)

    # Get the solver.
    with montblanc.rime_solver(slvr_cfg) as slvr:

        print slvr.nsrc

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

        with slvr.context as ctx:
            print slvr.vis_gpu.get()
        print slvr
