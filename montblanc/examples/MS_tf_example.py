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

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    FitsBeamSourceProvider,
    MSSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import (SinkProvider,
    MSSinkProvider)

class RadioSourceProvider(SourceProvider):

    def name(self):
        return "TF example"

    def point_lm(self, context):
        lm = np.empty(context.shape, context.dtype)

        montblanc.log.info(context.array_schema.shape)
        montblanc.log.info(context.iter_args)


        (lt, ut), (lb, ub), (lc, uc) = context.dim_extents(
            'ntime', 'nbl', 'nchan')

        lm[:,0] = 0.0008
        lm[:,1] = 0.0036


        lm[:,:] = 0
        return lm

    def point_stokes(self, context):
        stokes = np.empty(context.shape, context.dtype)
        stokes[:,:,0] = 1
        stokes[:,:,1:4] = 0
        return stokes

    def point_alpha(self, context):
        return np.zeros(context.shape, context.dtype)

    def frequency(self, context):
        return np.full(context.shape, 1.415e9, context.dtype)

    def ref_frequency(self, context):
        ref_freq = np.empty(context.shape, context.dtype)
        ref_freq[:] = 1.415e9

        return ref_freq

class RimeSinkProvider(SinkProvider):
    def name(self):
        return 'Sink'

    def model_vis(self, context):
        montblanc.log.info(context.data.ravel()[0:128].reshape(-1,4))
        montblanc.log.info(context.data.mean())
        montblanc.log.info(context.data.sum())

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-b', '--beam',
        type=str, default='', help='Base beam filename')
    parser.add_argument('-np','--npsrc',dest='npsrc',
        type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc',
        type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc',
        type=int, default=0, help='Number of Sersic Sources')
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
    montblanc.log.setLevel(logging.DEBUG)
    [h.setLevel(logging.DEBUG) for h in montblanc.log.handlers]

    slvr_cfg = montblanc.rime_solver_cfg(
        ntime=10000, na=3000, nchan=128,
        mem_budget=1024*1024*1024,
        data_source=Options.DATA_SOURCE_DEFAULT,
        sources=montblanc.sources(point=args.npsrc,
            gaussian=args.ngsrc,
            sersic=args.nssrc),
        dtype='double',
        auto_correlations=args.auto_correlations,
        version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Manages measurement sets
        ms_mgr = MeasurementSetManager(args.msfile, slvr, slvr_cfg)

        source_provs = []
        # Read problem info from the MS, taking observed visibilities from MODEL_DAT
        source_provs.append(MSSourceProvider(ms_mgr, 'MODEL_DATA'))
        # Add a beam when you're ready
        #source_provs.append(FitsBeamSourceProvider('beam_$(corr)_$(reim).fits'))
        source_provs.append(RadioSourceProvider())

        sink_provs = []
        # Dump model visibilities into CORRECTED_DATA
        sink_provs.append(MSSinkProvider(ms_mgr, 'CORRECTED_DATA'))

        slvr.solve(source_providers=source_provs,
            sink_providers=sink_provs)
