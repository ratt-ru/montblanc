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
import pyrap.tables as pt

import montblanc
import montblanc.impl.common.loaders

from montblanc.config import (RimeSolverConfig as Options)

# Measurement Set string constants
UVW = 'UVW'
CHAN_FREQ = 'CHAN_FREQ'
NUM_CHAN = 'NUM_CHAN'
REF_FREQUENCY = 'REF_FREQUENCY'
ANTENNA1 = 'ANTENNA1'
ANTENNA2 = 'ANTENNA2'
DATA = 'DATA'
FLAG = 'FLAG'
FLAG_ROW = 'FLAG_ROW'
WEIGHT_SPECTRUM = 'WEIGHT_SPECTRUM'
WEIGHT = 'WEIGHT'
SIGMA_SPECTRUM = 'SIGMA_SPECTRUM'
SIGMA = 'SIGMA'

class MeasurementSetLoader(montblanc.impl.common.loaders.MeasurementSetLoader):
    def load(self, solver, slvr_cfg):
        """
        Load the Measurement Set
        """
        tm = self.tables['main']
        ta = self.tables['ant']
        tf = self.tables['freq']

        ntime, na, nbl, nbands, nchan = solver.dim_global_size(
            'ntime', 'na', 'nbl', 'nbands', 'nchan')

        # Transfer wavelengths
        wavelength = (montblanc.constants.C/tf.getcol(CHAN_FREQ)
            .reshape(solver.wavelength.shape)
            .astype(solver.wavelength.dtype))
        solver.transfer_wavelength(wavelength)

        # Transfer reference wavelengths
        ref_waves_per_band = np.concatenate(
            [np.repeat(montblanc.constants.C/rf, size) for rf, size
            in zip(tf.getcol(REF_FREQUENCY), tf.getcol(NUM_CHAN))], axis=0)
        solver.transfer_ref_wavelength(ref_waves_per_band)

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

        # If the main table has visibilities for multiple bands, then
        # there will be multiple (duplicate) UVW, ANTENNA1 and ANTENNA2 values
        # Ensure uniqueness to get a single value here
        uvw_table = pt.taql('SELECT FROM $tm ORDERBY UNIQUE TIME, ANTENNA1, ANTENNA2')

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = uvw_table.getcol(UVW)
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape,uvw_shape)

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1.

        # We choose u_0 = 0 and thus have
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2
        uvw=np.empty(shape=solver.uvw.shape, dtype=solver.uvw.dtype)
        uvw[:,:,1:na] = ms_uvw.reshape(file_uvw_shape).transpose(uvw_transpose) \
            .astype(solver.ft)[:,:,:na-1]
        uvw[:,:,0] = solver.ft(0)
        solver.transfer_uvw(np.ascontiguousarray(uvw))

        # Get the baseline antenna pairs and correct the axes
        ant1 = uvw_table.getcol(ANTENNA1).reshape(file_ant_shape)
        ant2 = uvw_table.getcol(ANTENNA2).reshape(file_ant_shape)
        if data_order == Options.DATA_ORDER_OTHER:
            ant1 = ant1.transpose(ant_transpose)
            ant2 = ant2.transpose(ant_transpose)

        expected_ant_shape = (ntime,nbl)

        assert expected_ant_shape == ant1.shape, \
            '{a} shape is {r} != expected {e}'.format(
                a=ANTENNA1, r=ant1.shape, e=expected_ant_shape)

        assert expected_ant_shape == ant2.shape, \
            '{a} shape is {r} != expected {e}'.format(
                a=ANTENNA2, r=ant2.shape, e=expected_ant_shape)


        solver.transfer_antenna1(np.ascontiguousarray(ant1))
        solver.transfer_antenna2(np.ascontiguousarray(ant2))

        uvw_table.close()

        # Load in visibility data, if it exists.
        if tm.colnames().count(DATA) > 0:
            montblanc.log.info('{lp} Loading visibilities '
                'into the {ovis} array'.format(
                    lp=self.LOG_PREFIX, ovis='observed_vis'))
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = tm.getcol('DATA').reshape(file_data_shape) \
                .transpose(data_transpose).astype(solver.ct)
            solver.transfer_observed_vis(np.ascontiguousarray(vis_data))
        else:
            montblanc.log.info('{lp} No visibilities found.'
                .format(lp=self.LOG_PREFIX))
            # Should be zeroed out by array defaults

        # Load in flag data if available
        if tm.colnames().count(FLAG) > 0:
            montblanc.log.info('{lp} Loading flag data.'.format(
                    lp=self.LOG_PREFIX))

            flag = tm.getcol(FLAG)
            flag_row = tm.getcol(FLAG_ROW)

            # Incorporate the flag_row data into the larger flag matrix
            flag = np.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

            # Reshape and transpose
            flag = flag.reshape(file_data_shape).transpose(
                data_transpose).astype(solver.flag.dtype)

            # Transfer, asking for contiguity
            solver.transfer_flag(np.ascontiguousarray(flag))
        else:
            montblanc.log.info('{lp} No flag data found.'
                .format(lp=self.LOG_PREFIX))
            # Should be zeroed out by array defaults

        # Should we initialise our weights from the MS data?
        init_weights = slvr_cfg.get(Options.INIT_WEIGHTS)
 
        chans_per_band = nchan // nbands

        # Load in weighting data, if it exists
        if init_weights is not Options.INIT_WEIGHTS_NONE:
            if init_weights == Options.INIT_WEIGHTS_WEIGHT:
            # Obtain weighting information from WEIGHT_SPECTRUM
            # preferably, otherwise WEIGHT.
                if tm.colnames().count(WEIGHT_SPECTRUM) > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    montblanc.log.info('{lp} Initialising {wv} from {n}.'
                        .format(lp=self.LOG_PREFIX, wv='weight_vector',
                            n=WEIGHT_SPECTRUM))

                    weight_vector = tm.getcol(WEIGHT_SPECTRUM)
                elif tm.colnames().count(WEIGHT) > 0:
                    # Otherwise we should try obtain the weightings from WEIGHT.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    montblanc.log.info('{lp} Initialising {wv} from {n}.'
                        .format(lp=self.LOG_PREFIX, wv='weight_vector',
                            n=WEIGHT))

                    weight = tm.getcol(WEIGHT)[:,np.newaxis,:]
                    weight_vector = weight*np.ones(shape=(chans_per_band,1))
                else:
                    # We couldn't find anything, set to one
                    montblanc.log.info('{lp} No {ws} or {w} columns. '
                        'Initialising {wv} from with ones.'
                        .format(lp=self.LOG_PREFIX, ws=WEIGHT_SPECTRUM,
                            w=WEIGHT, wv='weight_vector'))

                    weight_vector = np.ones(
                        shape=solver.weight_vector_shape,
                        dtype=solver.weight_vector_dtype)
            elif init_weights == Options.INIT_WEIGHTS_SIGMA:
                # Obtain weighting information from SIGMA_SPECTRUM
                # preferably, otherwise SIGMA.
                if tm.colnames().count(SIGMA_SPECTRUM) > 0:
                    # Try obtain the weightings from WEIGHT_SPECTRUM first.
                    montblanc.log.info('{lp} Initialising {wv} from {n}.'
                        .format(lp=self.LOG_PREFIX, wv='weight_vector',
                            n=SIGMA_SPECTRUM))

                    weight_vector = tm.getcol(SIGMA_SPECTRUM)
                elif tm.colnames().count(SIGMA) > 0:
                    # Otherwise we should try obtain the weightings from WEIGHT.
                    # This doesn't have per-channel weighting, so we introduce
                    # this with a broadcast
                    montblanc.log.info('{lp} Initialising {wv} from {n}.'
                        .format(lp=self.LOG_PREFIX, wv='weight_vector',
                            n=SIGMA))

                    sigma = tm.getcol(SIGMA)[:,np.newaxis,:]
                    weight_vector = weight*np.ones(shape=(chans_per_band,1))
                else:
                    # We couldn't find anything, set to one
                    montblanc.log.info('{lp} No {ss} or {s} columns. '
                        'Initialising {wv} from with ones.'
                        .format(lp=self.LOG_PREFIX, ss=SIGMA_SPECTRUM,
                            s=SIGMA, wv='weight_vector'))

                    weight_vector = np.ones(shape=solver.weight_vector_shape,
                        dtype=solver.weight_vector_dtype)
            else:
                raise Exception, 'init_weights used incorrectly!'

            assert weight_vector.shape == (ntime*nbl*nbands, chans_per_band, 4)

            weight_vector = weight_vector.reshape(file_data_shape) \
                .transpose(data_transpose).astype(solver.ft)

            solver.transfer_weight_vector(np.ascontiguousarray(weight_vector))

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)
