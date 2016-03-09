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

def repeat_brightness_over_time(slvr, parser):
    """
    Assuming our sky model file doesn't have
    [I Q U V alpha] values for each timestep,
    we need to replicate these values across
    each timestep. Figure out the time dimension
    of the the brightness array and repeat
    the array along it.
    """
    # Get the brightness matrix. In general, montblanc support
    # time-varying brightness, but it's not practical to specify
    # this in a sky model text file. So we assume a shape here
    # (5, nsrc) and call fiddle_brightness to replicate these
    # values across the time dimension
    ntime = slvr.dim_global_size('ntime')
    R = slvr.get_array_record('brightness')
    time_dim = R.sshape.index('ntime')
    no_time_shape = tuple([d for i, d in enumerate(R.shape) if i != time_dim])

    brightness = parser.shape_arrays(['I','Q','U','V','alpha'],
        no_time_shape, slvr.brightness_dtype)

    return np.repeat(brightness, ntime, time_dim).reshape(R.shape)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-s', '--sky-file', dest='sky_file', type=str, required=True, help='Sky Model File')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str, default='v2', choices=['v2','v3','v4','v5'],
        help='BIRO Pipeline Version.')

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
        store_cpu=False, version=args.version)

    with montblanc.rime_solver(slvr_cfg) as slvr:

        # Get the lm coordinates
        lm = sky_parse.shape_arrays(['l','m'], slvr.lm_shape, slvr.lm_dtype)

        # Get the brightness values, repeating them over time
        brightness = repeat_brightness_over_time(slvr, sky_parse)

        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.dim_global_size('ngsrc') > 0:
            gauss_shape = sky_parse.shape_arrays(['el','em','eR'],
                slvr.gauss_shape_shape, slvr.gauss_shape_dtype)

        # Create a bayesian model and upload it to the GPU
        bayes_data = mbu.random_like(slvr.bayes_data_gpu)

        # Generate random antenna pointing errors
        point_errors = mbu.random_like(slvr.point_errors_gpu)

        # Generate and transfer a noise vector.
        weight_vector = mbu.random_like(slvr.weight_vector_gpu)
        slvr.transfer_weight_vector(weight_vector)

        # Execute the pipeline
        for i in range(args.count):
            # Set data on the solver object. Uploads to GPU
            slvr.transfer_lm(lm)
            slvr.transfer_brightness(brightness)
            slvr.transfer_point_errors(point_errors)
            # Change parameters for this run
            slvr.set_sigma_sqrd((np.random.random(1)**2)[0])
            # Execute the pipeline
            slvr.solve()

            # The chi squared result is set on the solver object
            print 'Chi Squared Value', slvr.X2

            # Obtain the visibilities  (slow)
            #with slvr.context:
            #    V = slvr.vis_gpu.get()

        # Print information about the simulation
        print slvr
