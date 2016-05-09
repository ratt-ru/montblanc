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

from montblanc.config import (
    RimeSolverConfig as Options)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-v','--version',dest='version', type=str,
        default=Options.VERSION_FOUR, choices=Options.VALID_VERSIONS,
        help='RIME Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Get the solver.
    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=1, gaussian=0, sersic=0),
        dtype='double', version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        if args.version in [Options.VERSION_TWO]:
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            l, m = lm[0,:], lm[1,:]
            l[:] = 0.1
            m[:] = 0.25

            slvr.transfer_lm(lm)

            B = np.empty(shape=slvr.brightness.shape, dtype=slvr.brightness.dtype)
            I, Q, U, V, alpha = B[0,:,:], B[1,:,:], B[2,:,:], B[3,:,:], B[4,:,:]
            I[:] = 2
            Q[:] = 1
            U[:] = 1
            V[:] = 1
            alpha[:] = 0.5

            slvr.transfer_brightness(B)
        elif args.version in [Options.VERSION_FOUR]:
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            l, m = lm[:,0], lm[:,1]
            l[:] = 0.1
            m[:] = 0.25

            slvr.transfer_lm(lm)

            stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
            alpha = np.empty(shape=slvr.alpha.shape, dtype=slvr.alpha.dtype)
            I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
            I[:] = 2
            Q[:] = 1
            U[:] = 1
            V[:] = 1
            alpha[:] = 0.5
            slvr.transfer_stokes(stokes)
            slvr.transfer_alpha(alpha)
        elif args.version in [Options.VERSION_FIVE]:
            slvr.lm[:,0] = 0.1
            slvr.lm[:,1] = 0.25

            slvr.stokes[:,:,0] = 2
            slvr.stokes[:,:,1] = 1
            slvr.stokes[:,:,2] = 1
            slvr.stokes[:,:,3] = 1
            slvr.alpha[:] = 0.5

        slvr.solve()

        if args.version in [Options.VERSION_TWO]:
            vis = slvr.retrieve_model_vis().transpose(1, 2, 3, 0)
        elif args.version in [Options.VERSION_FOUR]:
            vis = slvr.retrieve_model_vis()
        elif args.version in [Options.VERSION_FIVE]:
            vis = slvr.model_vis

        print vis
        print vis.shape

        print slvr.X2
