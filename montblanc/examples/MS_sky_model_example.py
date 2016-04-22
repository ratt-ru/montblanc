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

from montblanc.config import (
    RimeSolverConfig as Options)

def repeat_brightness_over_time(slvr, parser):
    """
    Assuming our sky model file doesn't have
    [I Q U V alpha] values for each timestep,
    we need to replicate these values across
    each timestep. Figure out the time dimension
    of the the stokes and alpha array and repeat
    the array along it.
    """
    # Get the stokes matrix. In general, montblanc supports
    # time-varying stokes, but it's not practical to specify
    # this in a sky model text file, so we replicate these
    # values across the time dimension
    ntime = slvr.dim_global_size('ntime')

    stokes_record = slvr.array('stokes')
    time_dim = stokes_record.sshape.index('ntime')
    no_time_shape = tuple([d if i != time_dim else 1
        for i, d in enumerate(stokes_record.shape)])

    stokes = parser.shape_arrays(['I','Q','U','V'],
        no_time_shape, slvr.stokes_dtype)
    stokes = (np.repeat(stokes, ntime, time_dim)
        .reshape(stokes_record.shape))

    alpha_record = slvr.array('alpha')
    time_dim = alpha_record.sshape.index('ntime')
    no_time_shape = tuple([d if i != time_dim else 1
        for i, d in enumerate(alpha_record.shape)])

    alpha = parser.shape_arrays(['alpha'],
        no_time_shape, slvr.alpha_dtype)

    print alpha.shape, alpha_record.sshape
    alpha = (np.repeat(alpha, ntime, time_dim)
        .reshape(alpha_record.shape))

    return stokes, alpha

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-s', '--sky-file', dest='sky_file', type=str, required=True, help='Sky Model File')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str, default='v4', choices=[Options.VERSION_FOUR],
        help='RIME Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.WARN)

    # Parse the sky model file
    sky_parse = mbu.parse_sky_model(args.sky_file)

    sources = montblanc.sources(point=sky_parse.src_counts.get('npsrc', 0),
        gaussian=sky_parse.src_counts.get('ngsrc', 0),
        sersic=sky_parse.src_counts.get('nssrc', 0))

    slvr_cfg = montblanc.rime_solver_cfg(msfile=args.msfile,
        sources=sources, init_weights=None, weight_vector=False,
        version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:
        # Get the lm coordinates
        lm = sky_parse.shape_arrays(['l','m'], slvr.lm_shape, slvr.lm_dtype)

        # Get the stokes and alpha parameters
        stokes, alpha = repeat_brightness_over_time(slvr, sky_parse)

        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.dim_global_size('ngsrc') > 0:
            gauss_shape = sky_parse.shape_arrays(['el','em','eR'],
                slvr.gauss_shape_shape, slvr.gauss_shape_dtype)

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
            #with slvr.context:
            #    V = slvr.vis.get()

        # Print information about the simulation
        print slvr
