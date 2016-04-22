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
small enough to fit within a single GPU's memory. This caters for
small problem sizes where the user may wish to manually specify and
optimise data transfers to the GPU. It's also use for internal testing code.

In this use case data the user must manually transfer data to and
from the GPU, call the solver and then pull the data off. e.g.

slvr.transfer_lm(lm)             # Transfer lm coordinates
slvr.solve()                     # Execute the solver
vis = slvr.retrieve_model_vis()  # Retrieve the model visibilities

For larger problem sizes and more general use cases, please
see the single node example in this directory.
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
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc', type=int, default=0, help='Number of Sersic Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str, default=Options.VERSION_FOUR,
        choices=[Options.VERSION_TWO, Options.VERSION_FOUR], help='BIRO Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.INFO)

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=args.npsrc, gaussian=args.ngsrc, sersic=args.nssrc),
        init_weights='weight', weight_vector=False,
        dtype='double', version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Random point source coordinates in the l,m,n (brightness image) domain
        lm = mbu.random_like(slvr.lm)*0.1

        if args.version in [Options.VERSION_TWO]:
            # Random brightness matrix for the point sources
            brightness = mbu.random_like(slvr.brightness)
        elif args.version in [Options.VERSION_FOUR]:
            # Need a positive semi-definite brightness
            # matrix for v4 and v5
            stokes = np.empty(shape=slvr.stokes_shape, dtype=slvr.stokes_dtype)
            I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
            Q[:] = np.random.random(size=Q.shape)-0.5
            U[:] = np.random.random(size=U.shape)-0.5
            V[:] = np.random.random(size=V.shape)-0.5
            noise = np.random.random(size=(Q.shape))*0.1
            # Determinant of a brightness matrix
            # is I^2 - Q^2 - U^2 - V^2, noise ensures
            # positive semi-definite matrix
            I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)
            slvr.transfer_stokes(stokes)

            alpha = mbu.random_like(slvr.alpha)
            slvr.transfer_alpha(alpha)

        # E beam
        if args.version in [Options.VERSION_FOUR]:
            E_beam = mbu.random_like(slvr.E_beam)
            slvr.transfer_E_beam(E_beam)

        # G term
        if args.version in [Options.VERSION_FOUR]:
            G_term = mbu.random_like(slvr.G_term)
            slvr.transfer_G_term(G_term)

        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.dim_global_size('ngsrc') > 0:
            gauss_shape = mbu.random_like(slvr.gauss_shape)*0.1
            slvr.transfer_gauss_shape(gauss_shape)

        # Create observed visibilities and upload them to the GPU
        observed_vis = mbu.random_like(slvr.observed_vis)
        slvr.transfer_observed_vis(observed_vis)

        # Generate random antenna pointing errors
        point_errors = mbu.random_like(slvr.point_errors)

        # Generate and transfer a noise vector.
        weight_vector = mbu.random_like(slvr.weight_vector)
        slvr.transfer_weight_vector(weight_vector)

        # Execute the pipeline
        for i in range(args.count):
            # Set data on the solver object. Uploads to GPU
            slvr.transfer_lm(lm)
            if args.version in [Options.VERSION_TWO]:
                slvr.transfer_brightness(brightness)
            elif args.version in [Options.VERSION_FOUR]:
                slvr.transfer_stokes(stokes)
                slvr.transfer_alpha(alpha)
            slvr.transfer_point_errors(point_errors)
            # Change parameters for this run
            slvr.set_sigma_sqrd((np.random.random(1)**2)[0])
            # Execute the pipeline
            slvr.solve()

            # The chi squared result is set on the solver object
            print 'Chi Squared Value', slvr.X2


        # Obtain the visibilities 
        V = slvr.retrieve_model_vis()

        # Print information about the simulation
        print slvr
