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
    BiroSolverConfig as Options)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-v','--version',dest='version', type=str, default='v2', choices=['v2','v3', 'v4'],
        help='BIRO Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Get the solver.
    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=1, gaussian=0, sersic=0),
        dtype='double', version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        if args.version in [Options.VERSION_TWO, Options.VERSION_THREE]:
            lm = np.empty(shape=slvr.lm_shape, dtype=slvr.lm_dtype)
            l, m = lm[0,:], lm[1,:]
            l[:] = 0.1
            m[:] = 0.25

            slvr.transfer_lm(lm)

            B = np.empty(shape=slvr.brightness_shape, dtype=slvr.brightness_dtype)
            I, Q, U, V, alpha = B[0,:,:], B[1,:,:], B[2,:,:], B[3,:,:], B[4,:,:]
            I[:] = 2
            Q[:] = 1
            U[:] = 1
            V[:] = 1
            alpha[:] = 0.5

            slvr.transfer_brightness(B)
        elif args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
            lm = np.empty(shape=slvr.lm_shape, dtype=slvr.lm_dtype)
            l, m = lm[:,0], lm[:,1]
            l[:] = 0.1
            m[:] = 0.25

            slvr.transfer_lm(lm)

            stokes = np.empty(shape=slvr.stokes_shape, dtype=slvr.stokes_dtype)
            alpha = np.empty(shape=slvr.alpha_shape, dtype=slvr.alpha_dtype)
            I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
            I[:] = 2
            Q[:] = 1
            U[:] = 1
            V[:] = 1
            alpha[:] = 0.5
            slvr.transfer_stokes(stokes)
            slvr.transfer_alpha(alpha)

        slvr.solve()

        if args.version in [Options.VERSION_TWO, Options.VERSION_THREE]:
            vis = slvr.retrieve_vis().transpose(1, 2, 3, 0)
        elif args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
            vis = slvr.retrieve_vis()

        print vis
        print vis.shape

        print slvr.X2
