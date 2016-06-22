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

from montblanc.config import RimeSolverConfig as Options

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
    
        # Create 1Jy point sources
        if args.version == Options.VERSION_TWO:
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            lm[0,:] = 0  # Set all l = 0
            lm[1,:] = 0  # Set all m = 0
            slvr.transfer_lm(lm)

            brightness = np.empty(shape=slvr.brightness.shape, dtype=slvr.brightness.dtype)
            brightness[0,:,:] = 1
            brightness[1,:,:] = 0
            brightness[2,:,:] = 0
            brightness[3,:,:] = 0
            brightness[4,:,:] = 0
            slvr.transfer_brightness(brightness)            

            # Solve the RIME
            slvr.solve()    
            model_vis = slvr.retrieve_model_vis()
        elif args.version == Options.VERSION_FOUR:
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            lm[:,0] = 0  # Set all l = 0
            lm[:,1] = 0  # Set all m = 0
            slvr.transfer_lm(lm)

            stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
            stokes[:,:,0] = 1
            stokes[:,:,1] = 0
            stokes[:,:,2] = 0
            stokes[:,:,3] = 0
            slvr.transfer_stokes(stokes)
            slvr.transfer_alpha(np.zeros(shape=slvr.alpha.shape, dtype=slvr.alpha.dtype))

            # Solve the RIME
            slvr.solve()    
            model_vis = slvr.retrieve_model_vis()
        elif args.version == Options.VERSION_FIVE:
            slvr.lm[:,0] = 0  # Set all l = 0 for all sources
            slvr.lm[:,1] = 0  # Set all m = 0 for all sources

            slvr.stokes[:,:,0] = 1 # Set I = 1Jy for all sources at all times
            slvr.stokes[:,:,1] = 0 # Set Q = 0 for all sources at all times
            slvr.stokes[:,:,2] = 0 # Set U = 0 for all sources at all times
            slvr.stokes[:,:,3] = 0 # Set V = 0 for all sources at all times

            slvr.alpha[:,:] = 0  # Set spectral index to 0 for all sources at all times

            # Solve the RIME
            slvr.solve()    
            model_vis = slvr.model_vis


        print model_vis
        print slvr
