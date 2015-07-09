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

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-v','--version',dest='version', type=str, default='v2', choices=['v2','v3'],
        help='BIRO Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Get the solver.

    slvr_cfg = montblanc.biro_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=1, gaussian=0, sersic=0),
        dtype='double', version=args.version)

    with montblanc.get_biro_solver(slvr_cfg) as slvr:
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
