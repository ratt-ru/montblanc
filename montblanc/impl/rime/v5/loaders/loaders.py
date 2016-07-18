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

import os

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
NUM_CHAN='NUM_CHAN'
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

POSITION = 'POSITION'
PHASE_DIR = 'PHASE_DIR'

WEIGHT_VECTOR = 'weight_vector'

class AWeightVectorStrategy(object):
    """ Weight Vector Strategy """
    def __init__(self, loader, slvr):
        self.loader = loader
        self.table = loader.tables['main']
        self.slvr = slvr
        self.ntime, self.na, self.nbl, self.nchan, self.nbands, self.npol = \
            slvr.dim_global_size('ntime', 'na', 'nbl', 'nchan', 'nbands', 'npol')
        self.wv_view = slvr.weight_vector.reshape(
            self.ntime*self.nbl*self.nbands, -1, self.npol)

    def log_strategy(self):
        raise NotImplementedError()

    def load(self, startrow, nrow):
        raise NotImplementedError()

class NoWeightStrategy(AWeightVectorStrategy):
    """ Don't load weights from the Measurement Set """
    def __init__(self, loader, slvr):
        super(NoWeightStrategy, self).__init__(loader, slvr)

    def log_strategy(self):
        self.loader.log("'{wv}' will not be initialised."
            .format(wv=WEIGHT_VECTOR))

    def load(self, startrow, nrow):
        pass

class WeightStrategy(AWeightVectorStrategy):
    """ Load weights from the WEIGHT or SIGMA column """
    def __init__(self, loader, slvr, column):
        super(WeightStrategy, self).__init__(loader, slvr)
        self.column = column

    def log_strategy(self):
        self.loader.log_load(self.column, WEIGHT_VECTOR)    

    def load(self, startrow, nrow):
        """
        Weights apply over all channels.
        Read into buffer before broadcasting to the solver array.
        """
        wv_buffer = self.table.getcol(self.column, startrow=startrow, nrow=nrow)
        self.wv_view[startrow:startrow+nrow,:,:] = wv_buffer[:,np.newaxis,:]

class SpectrumStrategy(AWeightVectorStrategy):
    """ Load per channel weights from the WEIGHT_SPECTRUM or SIGMA_SPECTRUM column """
    def __init__(self, loader, slvr, column):
        super(SpectrumStrategy, self).__init__(loader, slvr)
        self.column = column

    def log_strategy(self):
        self.loader.log_load(self.column, WEIGHT_VECTOR)

    def load(self, startrow, nrow):
        """
        Weights apply per channel. Dump directly into solver array.
        """
        self.table.getcolnp(self.column,
            self.wv_view[startrow:startrow+nrow,:,:],
            startrow=startrow, nrow=nrow)

class OnesStrategy(AWeightVectorStrategy):
    """ Set weights to one """
    def __init__(self, loader, slvr):
        super(OnesStrategy, self).__init__(loader, slvr)

    def log_strategy(self):
        self.loader.log_load('Initialising {wv} to 1.'.format(wv=WEIGHT_VECTOR))

    def load(self, startrow, nrow):
        self.wv_view[:] = 1

class MeasurementSetLoader(montblanc.impl.common.loaders.MeasurementSetLoader):
    def log_load(self, ms_name, slvr_name):
        self.log("'{M}' will be loaded into '{S}'."
            .format(M=ms_name, S=slvr_name))

    def weight_vector_strategy(self, slvr, init_weights, column_names):
        if init_weights is Options.INIT_WEIGHTS_NONE:
            return NoWeightStrategy(self, slvr)
        elif init_weights == Options.INIT_WEIGHTS_WEIGHT:
            if column_names.count(WEIGHT_SPECTRUM) > 0:
                return SpectrumStrategy(self, slvr, WEIGHT_SPECTRUM)
            elif column_names.count(WEIGHT) > 0:
                return WeightStrategy(self, slvr, WEIGHT)
            else:
                return OnesStrategy(self, slvr)
        elif init_weights == Options.INIT_WEIGHTS_SIGMA:
            if column_names.count(SIGMA_SPECTRUM) > 0:
                return SpectrumStrategy(self, slvr, SIGMA_SPECTRUM)
            elif column_names.count(SIGMA) > 0:
                return WeightStrategy(self, slvr, SIGMA)
            else:
                return OnesStrategy(self, slvr)

    def load(self, solver, slvr_cfg):
        """
        Load the Measurement Set
        """
        tm = self.tables['main']
        ta = self.tables['ant']
        tf = self.tables['freq']
        tfi = self.tables['field']

        ntime, na, nbl, nchan, nbands, npol = solver.dim_global_size(
            'ntime', 'na', 'nbl', 'nchan', 'nbands', 'npol')

        self.log("Processing main table {n}.".format(
                    n=os.path.split(self.msfile)[1]))

        msrows = tm.nrows()
        column_names = tm.colnames()

        # Determine row increments in terms of a time increment
        # This is required for calculating per antenna UVW coordinates below
        time_inc = 1
        nblbands = nbl*nbands

        while time_inc*nblbands < 5000:
            time_inc *= 2

        row_inc = time_inc*nblbands

        self.log('Processing rows in increments of {ri} = '
            '{ti} timesteps x {nbl} baselines x {nb} bands.'.format(
                ri=row_inc, ti=time_inc, nbl=nbl, nb=nbands))

        # Optionally loaded data
        data_present = False
        flag_present = False

        # Set up our weight vector loading strategy
        weight_strategy = self.weight_vector_strategy(solver,
            slvr_cfg.get(Options.INIT_WEIGHTS), column_names)

        # Check for presence of visibilities
        if column_names.count(DATA) > 0:
            data_present = True
            self.log_load(DATA, 'observed_vis')

        # Check for the presence of flags
        if column_names.count(FLAG) > 0:
            flag_present = True
            self.log_load(FLAG, 'flag')

        weight_strategy.log_strategy()

        self.log_load(UVW, 'uvw')
        self.log_load(ANTENNA1, 'antenna1')
        self.log_load(ANTENNA2, 'antenna2')

        # Iterate over the main MS rows
        for start in xrange(0, msrows, row_inc):
            nrows = min(row_inc, msrows - start)
            end = start + nrows

            self.log('Loading rows {s} -- {e}.'.format(
                s=start, e=end))

            if data_present:
                # Dump visibility data straight into the observed visibility array
                observed_vis_view = solver.observed_vis.reshape(ntime*nbl*nbands, -1, npol)
                tm.getcolnp(DATA, observed_vis_view[start:end,:,:],
                    startrow=start, nrow=nrows)

            if flag_present:
                # getcolnp doesn't handle solver.flag's dtype of np.uint8
                # Read into buffer and copy solver array
                
                # Get per polarisation flagging data
                flag_buffer = tm.getcol(FLAG, startrow=start, nrow=nrows)

                # Incorporate per visibility flagging into the buffer
                flag_row = tm.getcol(FLAG_ROW, startrow=start, nrow=nrows)
                flag_buffer = np.logical_or(flag_buffer,
                    flag_row[:,np.newaxis,np.newaxis])

                # Take a view of the solver array and copy the buffer in
                flag_view = solver.flag.reshape(ntime*nbl*nbands, -1, npol)
                flag_view[start:end,:,:] = flag_buffer.astype(solver.flag.dtype)

            # Execute weight vector loading strategy
            weight_strategy.load(start, nrows)

        # If the main table has visibilities for multiple bands, then
        # there will be multiple (duplicate) UVW, ANTENNA1 and ANTENNA2 values
        # Ensure uniqueness to get a single value here
        uvw_table = pt.taql("SELECT TIME, UVW, ANTENNA1, ANTENNA2 "
            "FROM $tm ORDERBY UNIQUE TIME, ANTENNA1, ANTENNA2")
        msrows = uvw_table.nrows()
        time_inc = 1

        while time_inc*nbl < 5000:
            time_inc *= 2

        row_inc = time_inc*nbl

        for start in xrange(0, msrows, row_inc):
            nrows = min(row_inc, msrows - start)
            end = start + nrows
            t_start = start // nbl
            t_end = end // nbl

            ant_view = solver.antenna1.reshape(ntime*nbl)
            uvw_table.getcolnp(ANTENNA1, ant_view[start:end],
                startrow=start, nrow=nrows)

            ant_view = solver.antenna2.reshape(ntime*nbl)
            uvw_table.getcolnp(ANTENNA2, ant_view[start:end],
                startrow=start, nrow=nrows)

            # Read UVW coordinates into a buffer
            uvw_buffer = (uvw_table.getcol(UVW, startrow=start, nrow=nrows)
                .reshape(t_end - t_start, nbl, 3))

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
            solver.uvw[t_start:t_end,1:na,:] = uvw_buffer[:,:na-1,:]
            solver.uvw[:,0,:] = 0

        self.log('Computing parallactic angles')
        # Compute parallactic angles
        time_table = pt.taql('SELECT TIME FROM $tm ORDERBY UNIQUE TIME')
        times = time_table.getcol(TIME)
        ref_ant_position = ta.getcol(POSITION, startrow=0, nrow=1)[0]
        phase_dir = tfi.getcol(PHASE_DIR)[0][0]

        solver.parallactic_angles[:] = mbu.parallactic_angles(phase_dir, ref_ant_position, times)

        time_table.close()
        uvw_table.close()

        self.log("Processing frequency table {n}.".format(
            n=os.path.split(self.freqfile)[1]))

        # Offset of first channel in the band
        band_ch0 = 0

        # Iterate over each band
        for b, (rf, bs) in enumerate(zip(tf.getcol(REF_FREQUENCY), tf.getcol(NUM_CHAN))):
            # Transfer this band's frequencies into the solver's frequency array
            from_str = ''.join([CHAN_FREQ, '[{b}][0:{bs}]'.format(b=b, bs=bs)])
            to_str = 'frequency[{s}:{e}]'.format(s=band_ch0, e=band_ch0+bs)
            self.log_load(from_str, to_str)
            tf.getcellslicenp(CHAN_FREQ,
                solver.frequency[band_ch0:band_ch0+bs],
                rownr=b, blc=(-1), trc=(-1))

            # Repeat this band's reference frequency in the solver's
            # reference frequency array
            from_str = ''.join([REF_FREQUENCY, '[{b}] == {rf}'.format(b=b, rf=rf)])
            to_str = 'ref_frequency[{s}:{e}]'.format(s=band_ch0, e= band_ch0+bs)
            self.log_load(from_str, to_str)
            solver.ref_frequency[band_ch0:band_ch0+bs] = np.repeat(rf, bs)

            # Next band
            band_ch0 += bs

        self.log("Processing antenna table {n}.".format(
            n=os.path.split(self.freqfile)[1]))

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)