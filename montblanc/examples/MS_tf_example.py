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

import logging
import numpy as np

import montblanc
import montblanc.util as mbu

from montblanc.config import RimeSolverConfig as Options

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc',
        type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc',
        type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc',
        type=int, default=0, help='Number of Sersic Sources')
    parser.add_argument('-c','--count',dest='count',
        type=int, default=10, help='Number of Iterations')
    parser.add_argument('-ac','--auto-correlations',dest='auto_correlations',
        type=lambda v: v.lower() in ("yes", "true", "t", "1"),
        choices=[True, False], default=False,
        help='Handle auto-correlations')
    parser.add_argument('-v','--version',dest='version', type=str,
        default=Options.VERSION_TENSORFLOW,
        choices=[Options.VERSION_TENSORFLOW],
        help='RIME Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.INFO)

    slvr_cfg = montblanc.rime_solver_cfg(
        ntime=10000, na=3000, nchan=128,
        data_source=Options.DATA_SOURCE_DEFAULT,
        sources=montblanc.sources(point=args.npsrc,
            gaussian=args.ngsrc,
            sersic=args.nssrc),
        init_weights='weight', weight_vector=False,
        dtype='double', auto_correlations=args.auto_correlations,
        context='blah',
        version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        slvr.solve()
