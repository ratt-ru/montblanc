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

"""
This script demonstrates a Montblanc use case for a problem size
that fits within a single compute node's memory. This caters for
large problem sizes that must be subdivided and batched for
solving on GPU(s).

In this use case data the user must edit numpy arrays defining the
problem on the solver object. Results are also present in arrays
on the solver, most importantly, the model_vis array.

slvr.lm[:] = random()*0.1       # Set lm coordinates on numpy array
slvr.solve()                    # Execute the solver
slvr.model_vis[:]               # Inspect model visibilities
"""


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
    parser.add_argument('-v','--version',dest='version',
        type=str, default=Options.VERSION_FIVE,
        choices=[Options.VERSION_FIVE], help='RIME Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.INFO)

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=args.npsrc,
            gaussian=args.ngsrc, sersic=args.nssrc),
        init_weights='weight', weight_vector=False,
        dtype='float', auto_correlations=args.auto_correlations,
        version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Random point source coordinates in the l,m,n (brightness image) domain
        slvr.lm[:] = mbu.random_like(slvr.lm)*0.1

        # Need a positive semi-definite brightness matrix
        I, Q, U, V = slvr.stokes[:,:,0], slvr.stokes[:,:,1], slvr.stokes[:,:,2], slvr.stokes[:,:,3]
        Q[:] = np.random.random(size=Q.shape)-0.5
        U[:] = np.random.random(size=U.shape)-0.5
        V[:] = np.random.random(size=V.shape)-0.5
        noise = np.random.random(size=(Q.shape))*0.1
        # Determinant of a brightness matrix
        # is I^2 - Q^2 - U^2 - V^2, noise ensures
        # positive semi-definite matrix
        I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)
        slvr.alpha[:] = mbu.random_like(slvr.alpha)

        # E beam
        slvr.E_beam[:] = mbu.random_like(slvr.E_beam)

        # G term
        slvr.G_term[:] = mbu.random_like(slvr.G_term)

        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.dim_global_size('ngsrc') > 0:
            slvr.gauss.shape[:] = mbu.random_like(slvr.gauss.shape)*0.1

        # Create observed visibilities and upload them to the GPU
        slvr.observed_vis[:] = mbu.random_like(slvr.observed_vis)

        # Generate and transfer a noise vector.
        slvr.weight_vector[:] = mbu.random_like(slvr.weight_vector)

        # Execute the pipeline
        for i in range(args.count):
            # Change our pointing errors on each run
            slvr.point_errors[:] = mbu.random_like(slvr.point_errors)
            # Change sigma squared property for this run
            slvr.set_sigma_sqrd((np.random.random(1)**2)[0])
            # Execute the pipeline
            slvr.solve()

            # The chi squared result is set on the solver object
            print 'Chi Squared Value', slvr.X2

        # Inspect the the visibilities 
        slvr.model_vis[:]

        # Print information about the simulation
        print slvr
