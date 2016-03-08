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
import montblanc.impl.common.loaders

from montblanc.config import (BiroSolverConfigurationOptions as Options)

class MeasurementSetLoader(montblanc.impl.common.loaders.MeasurementSetLoader):
    def load(self, solver, slvr_cfg):
        """
        Load the Measurement Set
        """
        tm = self.tables['main']
        ta = self.tables['ant']
        tf = self.tables['freq']

        na, nbl, ntime, nchan = solver.na, solver.nbl, solver.ntime, solver.nchan

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = tm.getcol('UVW')
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape, uvw_shape)

        # Create per antenna UVW coordinates.
        # u_01 = u_0 - u_1
        # u_02 = u_0 - u_2
        # ...
        # u_0N = u_0 - U_N
        # where N = na - 1.

        # We choose u_0 = 0 and thus have
        # u_1 = -u_01
        # u_2 = -u_02
        # ...
        # u_N = -u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_2 - u_1
        uvw = np.empty(shape=solver.uvw_shape, dtype=solver.uvw_dtype)
        uvw[:,1:na,:] = -ms_uvw.reshape(ntime, nbl, 3)[:,:na-1,:] \
            .astype(solver.ft)
        uvw[:,0,:] = solver.ft(0)
        solver.transfer_uvw(np.ascontiguousarray(uvw))

        # Determine the frequencys
        freqs = tf.getcol('CHAN_FREQ')
        solver.transfer_frequency(freqs[0].astype(solver.ft))

        # First dimension also seems to be of size 1 here...
        # Divide speed of light by frequency to get the frequency here.
        solver.set_ref_freq(tf.getcol('REF_FREQUENCY')[0])

        # Get the baseline antenna pairs and correct the axes
        ant1 = tm.getcol('ANTENNA1').reshape(ntime,nbl)
        ant2 = tm.getcol('ANTENNA2').reshape(ntime,nbl)

        expected_ant_shape = (ntime,nbl)

        assert expected_ant_shape == ant1.shape, \
            'ANTENNA1 shape is %s != expected %s' % (ant1.shape,expected_ant_shape)

        assert expected_ant_shape == ant2.shape, \
            'ANTENNA2 shape is %s != expected %s' % (ant2.shape,expected_ant_shape)

        ant_pairs = np.vstack((ant1,ant2)).reshape(solver.ant_pairs_shape)

        # Transfer the uvw coordinates, antenna pairs and frequencys to the GPU
        solver.transfer_ant_pairs(np.ascontiguousarray(ant_pairs))

        # Load in visibility data, if it exists.
        if tm.colnames().count('DATA') > 0:
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = tm.getcol('DATA').reshape(ntime,nbl,nchan,4) \
                .astype(solver.ct)
            solver.transfer_bayes_data(np.ascontiguousarray(vis_data))

        # Should we initialise our weights from the MS data?
        init_weights = slvr_cfg.get(Options.INIT_WEIGHTS)

        # Load in flag and weighting data, if it exists
        if init_weights is not Options.INIT_WEIGHTS_NONE and \
            tm.colnames().count('FLAG') > 0:
            # Get the flag data. Data flagged as True should be ignored,
            # therefore we flip the values so that when we multiply flag
            # by the weights later, a False value zeroes out the weight
            flag = (tm.getcol('FLAG') == False)
            flag_row = (tm.getcol('FLAG_ROW') == False)

            # Incorporate the flag_row data into the larger
            # flag matrix
            flag = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

            if init_weights == Options.INIT_WEIGHTS_WEIGHT:
                # Obtain weighting information from WEIGHT_SPECTRUM
                # preferably, otherwise WEIGHtm.
                if tm.colnames().count('WEIGHT_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('WEIGHT_SPECTRUM')
                elif tm.colnames().count('WEIGHT') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHT.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (tm.getcol('WEIGHT')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(solver.ft)
            elif init_weights == Options.INIT_WEIGHTS_SIGMA:
                # Obtain weighting information from SIGMA_SPECTRUM
                # preferably, otherwise SIGMA.
                if tm.colnames().count('SIGMA_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('SIGMA_SPECTRUM')
                elif tm.colnames().count('SIGMA') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHt.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    weight_vector = flag * \
                        (tm.getcol('SIGMA')[:,np.newaxis,:]*np.ones(shape=(nchan,1)))
                else:
                    # Just use the boolean flags as weighting values
                    weight_vector = flag.astype(solver.ft)
            else:
                raise Exception, 'init_weights used incorrectly!'

            assert weight_vector.shape == (ntime*nbl, nchan, 4)

            weight_vector = weight_vector.reshape(ntime,nbl,nchan,4) \
                .astype(solver.ft)

            solver.transfer_weight_vector(np.ascontiguousarray(weight_vector))

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)