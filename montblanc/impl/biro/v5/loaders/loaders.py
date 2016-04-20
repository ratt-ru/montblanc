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

import montblanc
import montblanc.impl.common.loaders

from montblanc.config import (BiroSolverConfig as Options)

# Measurement Set string constants
UVW = 'UVW'
CHAN_FREQ = 'CHAN_FREQ'
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

WEIGHT_VECTOR = 'weight_vector'

class AWeightVectorStrategy(object):
    """ Weight Vector Strategy """
    def __init__(self, loader, slvr):
        self.loader = loader
        self.table = loader.tables['main']
        self.slvr = slvr
        self.ntime, self.na, self.nbl, self.nchan, self.npol = \
            slvr.dim_global_size('ntime', 'na', 'nbl', 'nchan', 'npol')
        self.wv_view = slvr.weight_vector.reshape(
            self.ntime*self.nbl, self.nchan, self.npol)

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
    def __init__(self, loader, slvr):
        super(SigmaSpectrumStrategy, self).__init__(loader, slvr)
        self.column = column

    def log_strategy(self):
        self.loader.log_load(self.column, WEIGHT_VECTOR)

    def load(self, startrow, nrow):
        """
        Weights apply per channel. Dump directly into solver array.
        """
        self.table.getcolnp(self.column,
            self.wv_view[startrow:endrow,:,:],
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
        self.log("'{M}' will be loaded into '{S}'"
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

        ntime, na, nbl, nchan, npol = solver.dim_global_size(
            'ntime', 'na', 'nbl', 'nchan', 'npol')

        self.log("Processing main table {n}.".format(
                    n=os.path.split(self.msfile)[1]))

        msrows = tm.nrows()
        column_names = tm.colnames()

        if msrows != ntime*nbl:
            raise ValueError('MeasurementSet rows {msr} not equal to '
                'ntime x nbl = {nt} x {nbl} = {t} ({ac})'
                    .format(msr=msrows, nt=ntime, nbl=nbl, t=ntime*nbl,
                        ac=('auto-correlated' if slvr.is_auto_correlated()
                            else 'no auto-correlations')))

        # Work out our row increments in terms of a time increment
        time_inc = 1

        while nbl*time_inc < 5000:
            time_inc *= 2

        row_inc = time_inc*nbl

        self.log('Processing rows in increments of {ri} = '
            '{ti} timesteps x {nbl} baselines.'.format(
                ri=row_inc, ti=time_inc, nbl=nbl))

        # Optionally loaded data
        data_present = False
        flag_present = False

        # Set up our weight vector loading strategy
        weight_strategy = self.weight_vector_strategy(solver,
            slvr_cfg.get(Options.INIT_WEIGHTS), column_names)

        self.log_load(UVW, 'uvw')

        # Check for presence of visibilities
        if column_names.count(DATA) > 0:
            data_present = True
            self.log_load(DATA, 'bayes_data')

        # Check for the presence of flags
        if column_names.count(FLAG) > 0:
            flag_present = True
            self.log_load(FLAG, 'flag')

        weight_strategy.log_strategy()

        self.log_load(ANTENNA1, 'ant_pair[0]')
        self.log_load(ANTENNA2, 'ant_pair[1]')

        # Iterate over the main MS rows
        for start in range(0, msrows, row_inc):
            nrows = min(row_inc, msrows - start)
            end = start + nrows
            t_start = start // nbl
            t_end = end // nbl

            self.log('Loading rows {s} -- {e}'.format(
                s=start, e=end))

            # Read UVW coordinates into a buffer
            uvw_buffer = (tm.getcol(UVW, startrow=start, nrow=nrows)
                .reshape(t_end - t_start, nbl, 3))

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
            solver.uvw[t_start:t_end,1:na,:] = -uvw_buffer[:,:na-1,:]
            solver.uvw[:,0,:] = 0

            if data_present:
                # Dump visibility data straight into the bayes data array
                bayes_data_view = solver.bayes_data.reshape(ntime*nbl, nchan, npol)
                tm.getcolnp(DATA, bayes_data_view[start:end,:,:],
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
                flag_view = solver.flag.reshape(ntime*nbl, nchan, npol)
                flag_view[start:end,:,:] = flag_buffer.astype(solver.flag_dtype)

            antenna_view = solver.ant_pairs[0,:,:].reshape(ntime*nbl)
            tm.getcolnp(ANTENNA1, antenna_view[start:end],
                startrow=start, nrow=nrows)

            antenna_view = solver.ant_pairs[1,:,:].reshape(ntime*nbl)
            tm.getcolnp(ANTENNA2, antenna_view[start:end],
                startrow=start, nrow=nrows)

            # Execute weight vector loading strategy
            weight_strategy.load(start, nrows)

        self.log("Processing frequency table {n}.".format(
            n=os.path.split(self.freqfile)[1]))

        # Load the frequencies for the first spectral window (rownr=0) only
        self.log_load(CHAN_FREQ+'[0]', 'frequency')
        tf.getcellslicenp(CHAN_FREQ, solver.frequency, rownr=0, blc=(-1), trc=(-1))

        # Load the reference frequency for the first spectral window only
        self.log_load(REF_FREQUENCY+'[0]', 'ref_freq')
        solver.set_ref_freq(tf.getcol(REF_FREQUENCY)[0])

    def __enter__(solver):
        return super(MeasurementSetLoader,solver).__enter__()

    def __exit__(solver, type, value, traceback):
        return super(MeasurementSetLoader,solver).__exit__(type,value,traceback)