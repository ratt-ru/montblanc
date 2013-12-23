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

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RIME MS test script')
    parser.add_argument('msfile', help='Measurement Set File')
    parser.add_argument('-np','--npsrc',dest='npsrc', type=int, default=10, help='Number of Point Sources')
    parser.add_argument('-ng','--ngsrc',dest='ngsrc', type=int, default=0, help='Number of Gaussian Sources')
    parser.add_argument('-c','--count',dest='count', type=int, default=10, help='Number of Iterations')
    parser.add_argument('-v','--version',dest='version', type=str, default='v2', choices=['v2','v3'],
        help='BIRO Pipeline Version.')

    args = parser.parse_args(sys.argv[1:])

    # Set the logging level
    montblanc.log.setLevel(logging.WARN)

    # Get the BIRO solver.
    # npsrc : number of point sources
    # ngsrc : number of gaussian sources
    # init_weights : (1) None (2) 'sigma' or (3) 'weight'. Either
    #   (1) do not initialise the weight vector, or
    #   (2) initialise from the MS 'SIGMA' tables, or
    #   (3) initialise from the MS 'WEIGHT' tables.
    # weight_vector : indicates whether a weight vector should be used to
    #   compute the chi squared or a single sigma squared value
    # store_cpu : indicates whether copies of the data passed into the
    #   shared data transfer_* methods should be stored on the shared data object
    with montblanc.get_biro_solver(args.msfile,
        npsrc=args.npsrc, ngsrc=args.ngsrc, init_weights=None,
        weight_vector=False, store_cpu=False, version=args.version) as slvr:

        # Random point source coordinates in the l,m,n (brightness image) domain
        l=slvr.ft(np.random.random(slvr.nsrc)*0.1)
        m=slvr.ft(np.random.random(slvr.nsrc)*0.1)
        lm=np.array([l,m], dtype=slvr.ft)

        # Random brightness matrix for the point sources
        fI=slvr.ft(np.ones((slvr.ntime,slvr.nsrc,)))
        fQ=slvr.ft(np.random.random(slvr.ntime*slvr.nsrc)*0.5).reshape(slvr.ntime,slvr.nsrc)
        fU=slvr.ft(np.random.random(slvr.ntime*slvr.nsrc)*0.5).reshape(slvr.ntime,slvr.nsrc)
        fV=slvr.ft(np.random.random(slvr.ntime*slvr.nsrc)*0.5).reshape(slvr.ntime,slvr.nsrc)
        alpha=slvr.ft(np.random.random(slvr.ntime*slvr.nsrc)*0.1).reshape(slvr.ntime,slvr.nsrc)
        brightness = np.array([fI,fQ,fU,fV,alpha], dtype=slvr.ft)

        # If there are gaussian sources, create their
        # shape matrix and transfer it.
        if slvr.ngsrc > 0:
            el = slvr.ft(np.random.random(slvr.ngsrc)*0.1)
            em = slvr.ft(np.random.random(slvr.ngsrc)*0.1)
            R = slvr.ft(np.random.random(slvr.ngsrc))
            gauss_shape = np.array([el,em,R],dtype=slvr.ft)
            slvr.transfer_gauss_shape(gauss_shape)

        # Create a bayesian model and upload it to the GPU
        nviselements = np.product(slvr.vis_shape)
        bayes_data = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
            .astype(slvr.ct).reshape(slvr.vis_shape)
        slvr.transfer_bayes_data(bayes_data)

        # Generate random antenna pointing errors
        point_errors = np.random.random(np.product(slvr.point_errors_shape))\
            .astype(slvr.ft).reshape((slvr.point_errors_shape))

        # Generate and transfer a noise vector.
        weight_vector = np.random.random(np.product(slvr.weight_vector_shape))\
            .astype(slvr.ft).reshape((slvr.weight_vector_shape))
        slvr.transfer_weight_vector(weight_vector)

        # Execute the pipeline
        for i in range(args.count):
            # Set data on the shared data object. Uploads to GPU
            slvr.transfer_lm(lm)
            slvr.transfer_brightness(brightness)
            slvr.transfer_point_errors(point_errors)
            # Change parameters for this run
            slvr.set_sigma_sqrd((np.random.random(1)**2)[0])
            # Execute the pipeline
            slvr.solve()

            # The chi squared result is set on the shared data object
            print 'Chi Squared Value', slvr.X2

            # Obtain the visibilities  (slow)
            #V = slvr.vis_gpu.get()

        # Print information about the simulation
        print slvr