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

        ntime, na, nbl, nchan = solver.dim_global_size('ntime', 'na', 'nbl', 'nchan')
        data_order = slvr_cfg.get(Options.DATA_ORDER, Options.DATA_ORDER_CASA)

        # Define transpose axes to convert file uvw order
        # to montblanc array shape: (ntime, nbl)
        if data_order == Options.DATA_ORDER_OTHER:
            file_uvw_shape = (nbl, ntime, 3)
            uvw_transpose = (2,1,0)
            file_ant_shape = (nbl, ntime)
            ant_transpose = (1,0)
            file_data_shape = (nbl, ntime, nchan,4)
            data_transpose = (3,1,0,2)
        elif data_order == Options.DATA_ORDER_CASA:
            file_uvw_shape = (ntime, nbl,  3)
            uvw_transpose = (2,0,1)
            file_ant_shape = (ntime, nbl)
            file_data_shape = (ntime, nbl, nchan,4)
            data_transpose = (3,0,1,2)
        else:
            raise ValueError('Invalid UVW ordering %s', uvw_order)

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = tm.getcol('UVW')
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape,uvw_shape)

        uvw=np.empty(shape=solver.uvw_shape, dtype=solver.uvw_dtype)
        uvw[:,:,1:na] = ms_uvw.reshape(file_uvw_shape).transpose(uvw_transpose) \
            .astype(solver.ft)[:,:,:na-1]
        uvw[:,:,0] = solver.ft(0)
        solver.transfer_uvw(np.ascontiguousarray(uvw))

        # Determine the wavelengths
        freqs = tf.getcol('CHAN_FREQ')
        wavelength = (montblanc.constants.C/freqs[0]).astype(solver.ft)
        solver.transfer_wavelength(wavelength)

        # First dimension also seems to be of size 1 here...
        # Divide speed of light by frequency to get the wavelength here.
        solver.set_ref_wave(montblanc.constants.C/tf.getcol('REF_FREQUENCY')[0])

        # Get the baseline antenna pairs and correct the axes
        ant1 = tm.getcol('ANTENNA1').reshape(file_ant_shape)
        ant2 = tm.getcol('ANTENNA2').reshape(file_ant_shape)
        if data_order == Options.DATA_ORDER_OTHER:
            ant1 = ant1.transpose(ant_transpose)
            ant2 = ant2.transpose(ant_transpose)

        expected_ant_shape = (ntime,nbl)

        assert expected_ant_shape == ant1.shape, \
            'ANTENNA1 shape is %s != expected %s' % (ant1.shape,expected_ant_shape)

        assert expected_ant_shape == ant2.shape, \
            'ANTENNA2 shape is %s != expected %s' % (ant2.shape,expected_ant_shape)

        ant_pairs = np.vstack((ant1,ant2)).reshape(solver.ant_pairs_shape)

        # Transfer the uvw coordinates, antenna pairs and wavelengths to the GPU
        solver.transfer_ant_pairs(np.ascontiguousarray(ant_pairs))

        # Load in visibility data, if it exists.
        if tm.colnames().count('DATA') > 0:
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = tm.getcol('DATA').reshape(file_data_shape) \
                .transpose(data_transpose).astype(solver.ct)
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
            # preferably, otherwise WEIGHT.
                if tm.colnames().count('WEIGHT_SPECTRUM') > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('WEIGHT_SPECTRUM')
                elif tm.colnames().count('WEIGHT') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHt.
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
                    # Try obtain the weightings from WEIGHT_SPECTRUM firstm.
                    # It has the same dimensions as 'FLAG'
                    weight_vector = flag*tm.getcol('SIGMA_SPECTRUM')
                elif tm.colnames().count('SIGMA') > 0:
                    # Otherwise we should try obtain the weightings from WEIGHtm.
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

            weight_vector = weight_vector.reshape(file_data_shape) \
                .transpose(data_transpose).astype(solver.ft)

            solver.transfer_weight_vector(np.ascontiguousarray(weight_vector))

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)
