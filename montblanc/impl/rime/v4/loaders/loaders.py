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
import montblanc.util as mbu
import montblanc.impl.common.loaders

from montblanc.config import (RimeSolverConfig as Options)

# Measurement Set string constants
TIME = 'TIME'
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

POSITION = "POSITION"
PHASE_DIR = 'PHASE_DIR'

class MeasurementSetLoader(montblanc.impl.common.loaders.MeasurementSetLoader):
    def load(self, solver, slvr_cfg):
        """
        Load the Measurement Set
        """
        tm = self.tables['main']
        ta = self.tables['ant']
        tf = self.tables['freq']
        tfi = self.tables['field']

        ntime, na, nbl, nbands, nchan = solver.dim_global_size(
            'ntime', 'na', 'nbl', 'nbands', 'nchan')

        # Transfer frequencies
        freqs = (tf.getcol(CHAN_FREQ)
            .reshape(solver.frequency.shape)
            .astype(solver.frequency.dtype))
        solver.transfer_frequency(np.ascontiguousarray(freqs))

        # Transfer reference frequencies
        ref_freqs = tf.getcol(REF_FREQUENCY).astype(solver.ref_frequency.dtype)
        num_chans = tf.getcol(NUM_CHAN)

        ref_freqs_per_band = np.concatenate(
            [np.repeat(rf, size) for rf, size
            in zip(ref_freqs, num_chans)], axis=0)
        solver.transfer_ref_frequency(ref_freqs_per_band)

        # If the main table has visibilities for multiple bands, then
        # there will be multiple (duplicate) UVW, ANTENNA1 and ANTENNA2 values
        # Ensure uniqueness to get a single value here
        uvw_table = pt.taql("SELECT TIME, UVW, ANTENNA1, ANTENNA2 "
            "FROM $tm ORDERBY UNIQUE TIME, ANTENNA1, ANTENNA2")

        # Check that we're getting the correct shape...
        uvw_shape = (ntime*nbl, 3)

        # Read in UVW
        # Reshape the array and correct the axes
        ms_uvw = uvw_table.getcol(UVW)
        assert ms_uvw.shape == uvw_shape, \
            'MS UVW shape %s != expected %s' % (ms_uvw.shape, uvw_shape)

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
        uvw = np.empty(shape=solver.uvw.shape, dtype=solver.uvw.dtype)
        uvw[:,1:na,:] = ms_uvw.reshape(ntime, nbl, 3)[:,:na-1,:] \
            .astype(solver.ft)
        uvw[:,0,:] = solver.ft(0)
        solver.transfer_uvw(np.ascontiguousarray(uvw))

        # Get the baseline antenna pairs and correct the axes
        ant1 = uvw_table.getcol(ANTENNA1).reshape(ntime,nbl)
        ant2 = uvw_table.getcol(ANTENNA2).reshape(ntime,nbl)

        solver.transfer_antenna1(np.ascontiguousarray(ant1))
        solver.transfer_antenna2(np.ascontiguousarray(ant2))

        # Compute parallactic angles
        time_table = pt.taql('SELECT TIME FROM $tm ORDERBY UNIQUE TIME')
        times = time_table.getcol(TIME)
        antenna_positions = ta.getcol(POSITION)
        phase_dir = tfi.getcol(PHASE_DIR)[0][0]

        # Handle negative right ascension
        if phase_dir[0] < 0:
            phase_dir[0] += 2*np.pi

        parallactic_angles = mbu.parallactic_angles(phase_dir,
            antenna_positions, times)
        solver.transfer_parallactic_angles(parallactic_angles.astype(solver.parallactic_angles.dtype))

        time_table.close()
        uvw_table.close()

        # Load in visibility data, if it exists.
        if tm.colnames().count(DATA) > 0:
            montblanc.log.info('{lp} Loading visibilities '
                'into the {ovis} array'.format(
                    lp=self.LOG_PREFIX, ovis='observed_vis'))
            # Obtain visibilities stored in the DATA column
            # This comes in as (ntime*nbl,nchan,4)
            vis_data = (tm.getcol(DATA).reshape(solver.observed_vis.shape)
                .astype(solver.ct))
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

            # Reshape
            flag = flag.reshape(solver.flag.shape).astype(solver.flag.dtype)

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
                        shape=solver.weight_vector.shape,
                        dtype=solver.weight_vector.dtype)
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
                    weight_vector = (sigma*np.ones(shape=(chans_per_band,1)))
                else:
                    # We couldn't find anything, set to one
                    montblanc.log.info('{lp} No {ss} or {s} columns. '
                        'Initialising {wv} from with ones.'
                        .format(lp=self.LOG_PREFIX, ss=SIGMA_SPECTRUM,
                            s=SIGMA, wv='weight_vector'))

                    weight_vector = np.ones(shape=solver.weight_vector.shape,
                        dtype=solver.weight_vector.dtype)
            else:
                raise Exception, 'init_weights used incorrectly!'

            assert weight_vector.shape == (ntime*nbl*nbands, chans_per_band, 4)

            weight_vector = weight_vector.reshape(ntime,nbl,nchan,4) \
                .astype(solver.ft)

            solver.transfer_weight_vector(np.ascontiguousarray(weight_vector))

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)
