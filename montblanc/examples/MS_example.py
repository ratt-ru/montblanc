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

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-ns','--nssrc',dest='nssrc', type=int, default=0, help='Number of Sersic Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str, default='v2', choices=['v2','v3','v4','v5'],
        help='BIRO Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.WARN)

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=montblanc.sources(point=args.npsrc, gaussian=args.ngsrc, sersic=args.nssrc),
        init_weights=None, weight_vector=False, store_cpu=False,
        dtype='double', version=args.version)

    import json
    print json.dumps(slvr_cfg)

    with montblanc.rime_solver(slvr_cfg) as slvr:

        # Random point source coordinates in the l,m,n (brightness image) domain
        lm = (np.random.random(size=slvr.lm_shape) * 0.1) \
            .astype(slvr.lm_dtype)

        if args.version in [Options.VERSION_TWO, Options.VERSION_THREE]:
            # Random brightness matrix for the point sources
            brightness = np.random.random(
                    size=slvr.brightness_shape) \
                .astype(slvr.brightness_dtype)
        elif args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
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

            alpha = np.random.random(
                    size=slvr.alpha_shape) \
                .astype(slvr.alpha_dtype)
            slvr.transfer_alpha(alpha)

        # E beam
        if args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
            E_beam = (np.random.random(size=slvr.E_beam_shape) +
                np.random.random(size=slvr.E_beam_shape)*1j).astype(slvr.E_beam_dtype)
            slvr.transfer_E_beam(E_beam)

        # G term
        if args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
            G_term = (np.random.random(size=slvr.G_term_shape) +
                np.random.random(size=slvr.G_term_shape)*1j).astype(slvr.G_term_dtype)
            slvr.transfer_G_term(G_term)


        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.ngsrc > 0:
            gauss_shape = (np.random.random(
                    size=slvr.gauss_shape_shape) * 0.1) \
                .astype(slvr.gauss_shape_dtype)
            slvr.transfer_gauss_shape(gauss_shape)

        # Create a bayesian model and upload it to the GPU
        bayes_data = (np.random.random(size=slvr.bayes_data_shape) +
                np.random.random(size=slvr.bayes_data_shape)*1j) \
            .astype(slvr.bayes_data_dtype)

        # Generate random antenna pointing errors
        point_errors = np.random.random(
                size=slvr.point_errors_shape) \
            .astype(slvr.point_errors_dtype)

        # Generate and transfer a noise vector.
        weight_vector = np.random.random(
                size=slvr.weight_vector_shape) \
            .astype(slvr.ft)
        slvr.transfer_weight_vector(weight_vector)

        # Execute the pipeline
        for i in range(args.count):
            # Set data on the solver object. Uploads to GPU
            slvr.transfer_lm(lm)
            if args.version in [Options.VERSION_TWO, Options.VERSION_THREE]:
                slvr.transfer_brightness(brightness)
            elif args.version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
                slvr.transfer_stokes(stokes)
                slvr.transfer_alpha(alpha)
            slvr.transfer_point_errors(point_errors)
            # Change parameters for this run
            slvr.set_sigma_sqrd((np.random.random(1)**2)[0])
            # Execute the pipeline
            slvr.solve()

            # The chi squared result is set on the solver object
            print 'Chi Squared Value', slvr.X2

            # Obtain the visibilities  (slow)
            #V = slvr.vis_gpu.get()

        # Print information about the simulation
        print slvr
