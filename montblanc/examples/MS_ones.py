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
import montblanc.util as mbu

from montblanc.config import BiroSolverConfig as Options

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=1, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str,
        default=Options.VERSION_FOUR, choices=Options.VALID_VERSIONS, help='version')

    args = parser.parse_args(sys.argv[1:])

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=args.npsrc, gaussian=args.ngsrc, sersic=args.nssrc),
        dtype='double', version=args.version)

    # Get the solver.
    with montblanc.rime_solver(slvr_cfg) as slvr:
        nsrc = slvr.dim_global_size('nsrc')
        # Create point sources at zeros
        l=slvr.ft(np.zeros(nsrc))
        m=slvr.ft(np.zeros(nsrc))
        lm=mbu.shape_list([l,m], shape=slvr.lm_shape, dtype=slvr.lm_dtype)

        slvr.transfer_lm(lm)

        # Create 1Jy point sources
        if args.version in [Options.VERSION_TWO]:
            brightness = np.empty(shape=slvr.brightness_shape, dtype=slvr.brightness_dtype)
            brightness[0,:,:] = 1
            brightness[1,:,:] = 0
            brightness[2,:,:] = 0
            brightness[3,:,:] = 0
            brightness[4,:,:] = 0
            slvr.transfer_brightness(brightness)            
        elif args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
            stokes = np.empty(shape=slvr.stokes_shape, dtype=slvr.stokes_dtype)
            stokes[:,:,0] = 1
            stokes[:,:,1] = 0
            stokes[:,:,2] = 0
            stokes[:,:,3] = 0
            slvr.transfer_stokes(stokes)
            slvr.transfer_alpha(np.zeros(shape=slvr.alpha_shape, dtype=slvr.alpha_dtype))
    
        # Solve the RIME
        slvr.solve()

        with slvr.context as ctx:
            print slvr.vis.get()
        print slvr
